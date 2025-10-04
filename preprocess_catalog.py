#!/usr/bin/env python3
"""preprocess_catalog.py

Minimal, validated implementation to preprocess a CSV catalog and offer two
selection UIs: a Matplotlib-based selector and a tiny browser-based fallback.

This file is intentionally compact and avoids embedding large JS with Python
format braces by using string.Template for substitution.
"""

from __future__ import annotations
import argparse
import json
import os
import time
import webbrowser
import threading
import http.server
import socketserver
from string import Template
from typing import Optional, Dict

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Prefer TkAgg on macOS when tkinter is available
try:
    if os.sys.platform == 'darwin':
        import tkinter as _tk  # noqa: F401
        matplotlib.use('TkAgg', force=True)
except Exception:
    try:
        matplotlib.use('MacOSX', force=True)
    except Exception:
        pass


def preprocess_catalog(input_file: str, output_file: str, config: Optional[Dict] = None) -> bool:
    config = config or {}
    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        print(f"Failed to read {input_file}: {e}")
        return False

    # normalize column names to lower-case to avoid duplicates/case mismatches
    df.columns = [c.lower() for c in df.columns]

    # normalize common column name variants
    rename_map = {
        'lon': 'longitude', 'long': 'longitude', 'lng': 'longitude',
        'latitude_deg': 'latitude', 'lat': 'latitude',
        'mag': 'magnitude', 'Mw': 'magnitude', 'ML': 'magnitude', 'ml': 'magnitude', 'mb': 'magnitude',
        'time_utc': 'time', 'datetime': 'time', 'date': 'time', 'event_time': 'time', 'origin_time': 'time',
        'depth_km': 'depth', 'depth_m': 'depth', 'dep': 'depth', 'z': 'depth'
    }
    # apply renaming where possible (case-insensitive mapping)
    lower_map = {k.lower(): v for k, v in rename_map.items()}
    cols_map = {}
    for c in df.columns:
        mapped = lower_map.get(c.lower())
        if mapped:
            cols_map[c] = mapped
    if cols_map:
        df = df.rename(columns=cols_map)

    # If renaming produced duplicate column names, coalesce them (take first non-null per row)
    # Build a new DataFrame with unique column labels in original order
    cols = list(df.columns)
    seen = set()
    new_cols = []
    for c in cols:
        if c not in seen:
            new_cols.append(c)
            seen.add(c)
    if len(new_cols) != len(cols):
        new_df = pd.DataFrame(index=df.index)
        for name in new_cols:
            # find all original columns that equal this name
            dup_cols = [c for c in cols if c == name]
            if len(dup_cols) == 1:
                new_df[name] = df[dup_cols[0]]
            else:
                # take first non-null across duplicates
                new_df[name] = df[dup_cols].bfill(axis=1).iloc[:, 0]
        df = new_df

    # required spatial/magnitude/depth columns (time is optional)
    required_must = ['longitude', 'latitude', 'magnitude', 'depth']
    missing_must = [c for c in required_must if c not in df.columns]
    if missing_must:
        print(f"Missing required columns: {missing_must}")
        return False

    # Try to find/parse a time column if present or construct one from parts
    if 'time' not in df.columns:
        # look for any column with date/time-like name
        candidates = [c for c in df.columns if any(k in c.lower() for k in ['time', 'date', 'datetime'])]
        if candidates:
            df['time'] = pd.to_datetime(df[candidates[0]], errors='coerce')
            print(f"Parsed time from column: {candidates[0]}")
        else:
            # try to assemble from year/month/day (+ optional hour/min/sec)
            parts = {col.lower(): col for col in df.columns}
            if all(k in parts for k in ('year', 'month', 'day')):
                year = df[parts['year']].astype(int)
                month = df[parts['month']].astype(int)
                day = df[parts['day']].astype(int)
                hour = df[parts.get('hour', parts.get('hh'))].astype(int) if ('hour' in parts or 'hh' in parts) else 0
                minute = df[parts.get('minute', parts.get('mm'))].astype(int) if ('minute' in parts or 'mm' in parts) else 0
                second = df[parts.get('second', parts.get('ss'))].astype(int) if ('second' in parts or 'ss' in parts) else 0
                try:
                    df['time'] = pd.to_datetime(dict(year=year, month=month, day=day, hour=hour, minute=minute, second=second), errors='coerce')
                    print('Constructed time from year/month/day columns')
                except Exception:
                    df['time'] = pd.NaT
            else:
                df['time'] = pd.NaT
                print('No time column found; continuing with NaT times')

    # ensure time parsed where possible; don't fail if time is missing but ensure core columns exist
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'], errors='coerce')
    # drop rows missing core spatial/magnitude/depth info
    df = df.dropna(subset=['longitude', 'latitude', 'magnitude'])

    try:
        med = df['depth'].median()
        if not pd.isna(med) and med > 100:
            df['depth'] = df['depth'] / 1000.0
            print('Converted depth from meters to kilometers')
    except Exception:
        pass

    if config.get('min_magnitude') is not None:
        df = df[df['magnitude'] >= float(config['min_magnitude'])]
    if config.get('max_depth') is not None:
        df = df[df['depth'] <= float(config['max_depth'])]

    df = df.sort_values('time' if 'time' in df.columns else df.columns[0]).reset_index(drop=True)
    out_cols = ['longitude', 'latitude', 'magnitude', 'depth']
    if 'time' in df.columns:
        out_cols = ['time'] + out_cols
    out = df[[c for c in out_cols if c in df.columns]].copy()
    try:
        out.to_csv(output_file, index=False)
        print(f"Saved processed catalog to {output_file} ({len(out)} events)")
        return True
    except Exception as e:
        print(f"Failed to save processed catalog: {e}")
        return False


class InteractiveBoundingBox:
    def __init__(self, ax):
        from matplotlib.widgets import RectangleSelector, Button
        from matplotlib.patches import Polygon

        self.ax = ax
        self.poly = None
        self.center = None
        self.w = None
        self.h = None
        self.rotation = 0.0
        self._result = None

        self.selector = RectangleSelector(ax, self._onselect, drawtype='box', useblit=True,
                                          button=[1], spancoords='data', interactive=True)

        fig = ax.figure
        fig.subplots_adjust(bottom=0.18)
        ax_l = fig.add_axes([0.72, 0.03, 0.06, 0.05])
        ax_r = fig.add_axes([0.79, 0.03, 0.06, 0.05])
        ax_a = fig.add_axes([0.86, 0.03, 0.06, 0.05])
        self.btn_l = Button(ax_l, 'R-')
        self.btn_r = Button(ax_r, 'R+')
        self.btn_a = Button(ax_a, 'Apply')
        self.btn_l.on_clicked(self.rotate_left)
        self.btn_r.on_clicked(self.rotate_right)
        self.btn_a.on_clicked(self._on_apply)

    def _onselect(self, eclick, erelease):
        if eclick.xdata is None or erelease.xdata is None:
            return
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        self.center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        self.w = abs(x2 - x1)
        self.h = abs(y2 - y1)
        self._draw()

    def _draw(self):
        from matplotlib.patches import Polygon
        if self.poly is not None:
            try:
                self.poly.remove()
            except Exception:
                pass
            self.poly = None
        if self.center is None:
            self.ax.figure.canvas.draw_idle()
            return
        cx, cy = self.center
        corners = np.array([[-self.w / 2, -self.h / 2], [self.w / 2, -self.h / 2], [self.w / 2, self.h / 2], [-self.w / 2, self.h / 2]])
        th = np.deg2rad(self.rotation)
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        rot = corners.dot(R.T) + np.array([cx, cy])
        self.poly = self.ax.add_patch(Polygon(rot, closed=True, edgecolor='red', facecolor='red', alpha=0.2))
        self.ax.figure.canvas.draw_idle()

    def rotate_left(self, evt=None):
        if self.center is None:
            return
        self.rotation -= 15
        self._draw()

    def rotate_right(self, evt=None):
        if self.center is None:
            return
        self.rotation += 15
        self._draw()

    def _on_apply(self, evt=None):
        self._result = self.apply()

    def apply(self):
        if self.center is None:
            return None
        cx, cy = self.center
        return {
            'lon_min': cx - self.w / 2, 'lon_max': cx + self.w / 2,
            'lat_min': cy - self.h / 2, 'lat_max': cy + self.h / 2,
            'rotation': float(self.rotation)
        }


def plot_catalog_interactive(df, title: str = "Catalog"):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df['longitude'], df['latitude'], c=df['magnitude'], cmap='viridis', s=12)
    ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
    ax.set_title(title)
    bbox = InteractiveBoundingBox(ax)
    print('Interactive window open: draw rectangle then click Apply (use buttons)')
    plt.show()
    return bbox._result or bbox.apply()


def plot_catalog_web(df, timeout=300, port: Optional[int] = None, selected_out: Optional[str] = None):
    points = df[['longitude', 'latitude', 'magnitude']].to_dict(orient='records')
    lon_min, lon_max = float(df['longitude'].min()), float(df['longitude'].max())
    lat_min, lat_max = float(df['latitude'].min()), float(df['latitude'].max())

    js_points = json.dumps(points)
    
    # Use simple string replacement instead of Template to avoid JS brace conflicts  
    html_part1 = r"""
<!doctype html>
<html>
<head>
    <meta charset="utf-8">
    <title>Catalog Selector</title>
    <style>
        html,body{height:100%;margin:0;padding:0;font-family:sans-serif}
        #map{width:70%;height:100%;float:left;background:#e6f3ff;position:relative;overflow:hidden}
        #controls{width:30%;height:100%;float:right;padding:20px;box-sizing:border-box;overflow-y:auto}
        .earthquake{position:absolute;border-radius:50%;border:1px solid #333;background:red;opacity:0.7;cursor:pointer;z-index:10}
        .earthquake.selected{background:lime;border:2px solid #000}
        #selection-box{position:absolute;border:2px dashed blue;background:rgba(0,0,255,0.1);display:none;z-index:5;cursor:move}
        #selection-box.persistent{display:block}
        .selection-handle{position:absolute;width:8px;height:8px;background:blue;border:1px solid #fff;cursor:pointer;z-index:15}
        .handle-rotate{width:12px;height:12px;background:orange;border-radius:50%;cursor:crosshair}
        button{padding:10px;margin:5px;font-size:14px;cursor:pointer}
        input[type=range]{width:100%}
        .info{margin:10px 0;padding:10px;background:#f9f9f9;border:1px solid #ddd}
                .basemap{position:absolute;top:0;left:0;width:100%;height:100%;z-index:1;pointer-events:none;background:#f8f8f8}
        .gridline{stroke:#ddd;stroke-width:0.5;opacity:0.7}
        .gridlabel{font-size:10px;font-family:Arial,sans-serif;fill:#666;text-anchor:middle;dominant-baseline:middle;pointer-events:none}
    </style>
</head>
<body>
    <div id="map">
        <svg class="basemap" id="basemap"></svg>
        <div id="selection-box">
            <div class="selection-handle handle-rotate" style="top:-20px;left:50%;margin-left:-6px" title="Drag to rotate"></div>
        </div>
    </div>
    <div id="controls">
        <h3>Catalog Selector</h3>
        <div class="info">
            <strong>Total Events:</strong> <span id="total-count">0</span><br>
            <strong>Selected:</strong> <span id="selected-count">0</span>
        </div>
        <div class="info">
            <strong>Instructions:</strong><br>
            1. Click and drag to select earthquakes<br>
            2. Use rotation slider to rotate selection<br>
            3. Click Apply to submit selection
        </div>
        <div>
            <label>Rotation: <span id="rot-value">0</span>°</label><br>
            <input id="rot" type="range" min="-180" max="180" value="0">
        </div>
        <div>
            <button id="select-all">Select All</button>
            <button id="clear">Clear Selection</button>
        </div>
        <div>
            <button id="apply" style="background:#4CAF50;color:white;font-weight:bold">Apply Selection</button>
            <button id="cancel" style="background:#f44336;color:white">Cancel</button>
        </div>
    </div>
    <script>
        const pts = """
        
    html_part2 = r""";
        
        // Calculate bounds
        let minLat = Infinity, maxLat = -Infinity, minLon = Infinity, maxLon = -Infinity;
        pts.forEach(p => {
            minLat = Math.min(minLat, p.latitude);
            maxLat = Math.max(maxLat, p.latitude);
            minLon = Math.min(minLon, p.longitude);
            maxLon = Math.max(maxLon, p.longitude);
        });
        
        const latRange = maxLat - minLat;
        const lonRange = maxLon - minLon;
        const padding = 0.1; // 10% padding
        const displayMinLat = minLat - latRange * padding;
        const displayMaxLat = maxLat + latRange * padding;
        const displayMinLon = minLon - lonRange * padding;
        const displayMaxLon = maxLon + lonRange * padding;
        
        // Map container
        const mapDiv = document.getElementById('map');
        let mapRect = mapDiv.getBoundingClientRect();
        
        // Create basemap
        function createBasemap() {
            const svg = document.getElementById('basemap');
            svg.innerHTML = '';
            
            const mapWidth = mapDiv.offsetWidth;
            const mapHeight = mapDiv.offsetHeight;
            
            // Create simple coordinate grid only
            const lonStep = (displayMaxLon - displayMinLon) > 10 ? 2 : 1;
            const latStep = (displayMaxLat - displayMinLat) > 10 ? 2 : 1;
            
            // Longitude lines (vertical)
            for (let lon = Math.ceil(displayMinLon / lonStep) * lonStep; lon <= displayMaxLon; lon += lonStep) {
                const x = (lon - displayMinLon) / (displayMaxLon - displayMinLon) * mapWidth;
                const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
                line.setAttribute('x1', x);
                line.setAttribute('y1', 0);
                line.setAttribute('x2', x);
                line.setAttribute('y2', mapHeight);
                line.setAttribute('class', 'gridline');
                svg.appendChild(line);
                
                // Label
                const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                text.setAttribute('x', x);
                text.setAttribute('y', mapHeight - 5);
                text.setAttribute('class', 'gridlabel');
                text.textContent = lon + '°';
                svg.appendChild(text);
            }
            
            // Latitude lines (horizontal)
            for (let lat = Math.ceil(displayMinLat / latStep) * latStep; lat <= displayMaxLat; lat += latStep) {
                const y = (displayMaxLat - lat) / (displayMaxLat - displayMinLat) * mapHeight;
                const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
                line.setAttribute('x1', 0);
                line.setAttribute('y1', y);
                line.setAttribute('x2', mapWidth);
                line.setAttribute('y2', y);
                line.setAttribute('class', 'gridline');
                svg.appendChild(line);
                
                // Label
                const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                text.setAttribute('x', 5);
                text.setAttribute('y', y - 5);
                text.setAttribute('class', 'gridlabel');
                text.textContent = lat + '°';
                svg.appendChild(text);
            }
        }
        
        // Create earthquake markers
        const markers = [];
        let selectedIndices = new Set();
        
        function latLonToPixel(lat, lon) {
            const mapWidth = mapDiv.offsetWidth;
            const mapHeight = mapDiv.offsetHeight;
            const x = ((lon - displayMinLon) / (displayMaxLon - displayMinLon)) * mapWidth;
            const y = ((displayMaxLat - lat) / (displayMaxLat - displayMinLat)) * mapHeight;
            return {x, y};
        }
        
        function pixelToLatLon(x, y) {
            const mapWidth = mapDiv.offsetWidth;
            const mapHeight = mapDiv.offsetHeight;
            const lon = displayMinLon + (x / mapWidth) * (displayMaxLon - displayMinLon);
            const lat = displayMaxLat - (y / mapHeight) * (displayMaxLat - displayMinLat);
            return {lat, lon};
        }
        
        function createMarkers() {
            console.log('createMarkers called');
            console.log('Map div dimensions:', mapDiv.offsetWidth, 'x', mapDiv.offsetHeight);
            
            // Clear existing markers
            markers.forEach(m => m.element.remove());
            markers.length = 0;
            
            if (!pts || pts.length === 0) {
                console.error('No points data available');
                return;
            }
            
            console.log('Creating markers for all', pts.length, 'earthquakes');
            console.log('Display bounds:', displayMinLon, displayMaxLon, displayMinLat, displayMaxLat);
            
            // Create markers for ALL earthquakes, but optimize rendering
            function createBatch(startIdx) {
                const batchSize = 1000;
                const endIdx = Math.min(startIdx + batchSize, pts.length);
                
                for (let idx = startIdx; idx < endIdx; idx++) {
                    const pt = pts[idx];
                    const {x, y} = latLonToPixel(pt.latitude, pt.longitude);
                    
                    const marker = document.createElement('div');
                    marker.className = 'earthquake';
                    marker.style.left = (x - 3) + 'px';
                    marker.style.top = (y - 3) + 'px';
                    marker.style.width = '6px';
                    marker.style.height = '6px';
                    marker.title = `Mag ${pt.magnitude.toFixed(1)} at ${pt.latitude.toFixed(3)}, ${pt.longitude.toFixed(3)}`;
                    
                    marker.addEventListener('click', (e) => {
                        e.stopPropagation();
                        if (selectedIndices.has(idx)) {
                            selectedIndices.delete(idx);
                            marker.classList.remove('selected');
                        } else {
                            selectedIndices.add(idx);
                            marker.classList.add('selected');
                        }
                        updateCounts();
                    });
                    
                    mapDiv.appendChild(marker);
                    markers.push({element: marker, index: idx, lat: pt.latitude, lon: pt.longitude});
                }
                
                // Update progress
                document.getElementById('total-count').textContent = `Loading ${endIdx}/${pts.length}`;
                
                // Continue with next batch if not done
                if (endIdx < pts.length) {
                    setTimeout(() => createBatch(endIdx), 5);
                } else {
                    console.log(`Created ${markers.length} markers for all earthquakes`);
                    updateCounts();
                }
            }
            
            // Start creating markers in batches
            createBatch(0);
        }
        
        function updateCounts() {
            document.getElementById('total-count').textContent = pts.length;
            document.getElementById('selected-count').textContent = selectedIndices.size;
        }
        
        // Selection box functionality
        let isSelecting = false;
        let isDragging = false;
        let isRotating = false;
        let selectionStart = null;
        let dragStart = null;
        let rotateStart = null;
        let currentRotation = 0;
        const selectionBox = document.getElementById('selection-box');
        const rotateHandle = selectionBox.querySelector('.handle-rotate');
        
        let selectionBoxState = {
            left: 0, top: 0, width: 0, height: 0, 
            centerX: 0, centerY: 0, rotation: 0
        };
        
        function updateSelectionBox() {
            selectionBox.style.left = selectionBoxState.left + 'px';
            selectionBox.style.top = selectionBoxState.top + 'px';
            selectionBox.style.width = selectionBoxState.width + 'px';
            selectionBox.style.height = selectionBoxState.height + 'px';
            selectionBox.style.transform = `rotate(${selectionBoxState.rotation}deg)`;
            selectionBox.style.transformOrigin = 'center center';
        }
        
        function showSelectionBox() {
            selectionBox.classList.add('persistent');
            updateSelectionBox();
        }
        
        function hideSelectionBox() {
            selectionBox.classList.remove('persistent');
        }
        
        // New selection (drawing)
        mapDiv.addEventListener('mousedown', (e) => {
            if (e.target === mapDiv && !isDragging && !isRotating) {
                isSelecting = true;
                const rect = mapDiv.getBoundingClientRect();
                selectionStart = {x: e.clientX - rect.left, y: e.clientY - rect.top};
                selectionBoxState.left = selectionStart.x;
                selectionBoxState.top = selectionStart.y;
                selectionBoxState.width = 0;
                selectionBoxState.height = 0;
                selectionBoxState.rotation = 0;
                showSelectionBox();
            }
        });
        
        mapDiv.addEventListener('mousemove', (e) => {
            if (isSelecting) {
                const rect = mapDiv.getBoundingClientRect();
                const currentX = e.clientX - rect.left;
                const currentY = e.clientY - rect.top;
                
                selectionBoxState.left = Math.min(selectionStart.x, currentX);
                selectionBoxState.top = Math.min(selectionStart.y, currentY);
                selectionBoxState.width = Math.abs(currentX - selectionStart.x);
                selectionBoxState.height = Math.abs(currentY - selectionStart.y);
                selectionBoxState.centerX = selectionBoxState.left + selectionBoxState.width / 2;
                selectionBoxState.centerY = selectionBoxState.top + selectionBoxState.height / 2;
                
                updateSelectionBox();
            } else if (isDragging && dragStart) {
                const rect = mapDiv.getBoundingClientRect();
                const currentX = e.clientX - rect.left;
                const currentY = e.clientY - rect.top;
                
                const deltaX = currentX - dragStart.x;
                const deltaY = currentY - dragStart.y;
                
                selectionBoxState.left = dragStart.boxLeft + deltaX;
                selectionBoxState.top = dragStart.boxTop + deltaY;
                selectionBoxState.centerX = selectionBoxState.left + selectionBoxState.width / 2;
                selectionBoxState.centerY = selectionBoxState.top + selectionBoxState.height / 2;
                
                updateSelectionBox();
            } else if (isRotating && rotateStart) {
                const rect = mapDiv.getBoundingClientRect();
                const currentX = e.clientX - rect.left;
                const currentY = e.clientY - rect.top;
                
                const centerX = selectionBoxState.centerX;
                const centerY = selectionBoxState.centerY;
                
                const startAngle = Math.atan2(rotateStart.y - centerY, rotateStart.x - centerX);
                const currentAngle = Math.atan2(currentY - centerY, currentX - centerX);
                const deltaAngle = (currentAngle - startAngle) * 180 / Math.PI;
                
                selectionBoxState.rotation = rotateStart.rotation + deltaAngle;
                document.getElementById('rot').value = Math.round(selectionBoxState.rotation);
                document.getElementById('rot-value').textContent = Math.round(selectionBoxState.rotation);
                
                updateSelectionBox();
            }
        });
        
        mapDiv.addEventListener('mouseup', (e) => {
            if (isSelecting) {
                isSelecting = false;
                selectMarkersInBox();
            } else if (isDragging) {
                isDragging = false;
                dragStart = null;
                selectMarkersInBox();
            } else if (isRotating) {
                isRotating = false;
                rotateStart = null;
                selectMarkersInBox();
            }
        });
        
        // Selection box dragging
        selectionBox.addEventListener('mousedown', (e) => {
            if (e.target === selectionBox) {
                e.stopPropagation();
                isDragging = true;
                const rect = mapDiv.getBoundingClientRect();
                dragStart = {
                    x: e.clientX - rect.left,
                    y: e.clientY - rect.top,
                    boxLeft: selectionBoxState.left,
                    boxTop: selectionBoxState.top
                };
            }
        });
        
        // Rotation handle
        rotateHandle.addEventListener('mousedown', (e) => {
            e.stopPropagation();
            isRotating = true;
            const rect = mapDiv.getBoundingClientRect();
            rotateStart = {
                x: e.clientX - rect.left,
                y: e.clientY - rect.top,
                rotation: selectionBoxState.rotation
            };
        });
        
        function selectMarkersInBox() {
            if (selectionBoxState.width < 5 || selectionBoxState.height < 5) return;
            
            selectedIndices.clear();
            
            const centerX = selectionBoxState.centerX;
            const centerY = selectionBoxState.centerY;
            const rotation = selectionBoxState.rotation * Math.PI / 180;
            const cosR = Math.cos(-rotation);
            const sinR = Math.sin(-rotation);
            
            // Check ALL earthquake points, not just visible markers
            pts.forEach((pt, idx) => {
                const {x: markerX, y: markerY} = latLonToPixel(pt.latitude, pt.longitude);
                
                // Translate to box center
                const dx = markerX - centerX;
                const dy = markerY - centerY;
                
                // Rotate coordinates
                const rotatedX = dx * cosR - dy * sinR;
                const rotatedY = dx * sinR + dy * cosR;
                
                // Check if point is inside the unrotated box
                const halfWidth = selectionBoxState.width / 2;
                const halfHeight = selectionBoxState.height / 2;
                
                if (Math.abs(rotatedX) <= halfWidth && Math.abs(rotatedY) <= halfHeight) {
                    selectedIndices.add(idx);
                }
            });
            
            // Update visual selection on displayed markers
            markers.forEach(marker => {
                if (selectedIndices.has(marker.index)) {
                    marker.element.classList.add('selected');
                } else {
                    marker.element.classList.remove('selected');
                }
            });
            
            updateCounts();
            console.log(`Selected ${selectedIndices.size} earthquakes from total dataset`);
        }
        
        // Controls
        document.getElementById('rot').addEventListener('input', (e) => {
            selectionBoxState.rotation = parseFloat(e.target.value);
            document.getElementById('rot-value').textContent = Math.round(selectionBoxState.rotation);
            updateSelectionBox();
            selectMarkersInBox();
        });
        
        document.getElementById('select-all').addEventListener('click', () => {
            selectedIndices.clear();
            // Select ALL earthquakes in the dataset
            for (let i = 0; i < pts.length; i++) {
                selectedIndices.add(i);
            }
            // Update visual markers
            markers.forEach(marker => marker.element.classList.add('selected'));
            updateCounts();
            console.log(`Selected all ${selectedIndices.size} earthquakes`);
        });
        
        document.getElementById('clear').addEventListener('click', () => {
            selectedIndices.clear();
            markers.forEach(marker => marker.element.classList.remove('selected'));
            hideSelectionBox();
            updateCounts();
        });
        
        document.getElementById('apply').addEventListener('click', () => {
            if (selectedIndices.size === 0) {
                alert('Please select some earthquakes first');
                return;
            }
            
            const rotation = parseFloat(document.getElementById('rot').value || '0');
            const payload = {
                selected: Array.from(selectedIndices).sort((a, b) => a - b),
                rotation: rotation
            };
            
            fetch('/submit', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(payload)
            }).then(response => response.json())
            .then(data => {
                alert(`Selection submitted! Selected ${selectedIndices.size} earthquakes.`);
            }).catch(error => {
                console.error('Error:', error);
                alert('Error submitting selection');
            });
        });
        
        document.getElementById('cancel').addEventListener('click', () => {
            const payload = {cancel: true};
            fetch('/submit', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(payload)
            }).then(() => {
                alert('Selection cancelled');
            });
        });
        
        // Initialize - Optimized version
        console.log('Loading', pts ? pts.length : 0, 'earthquake points...');
        
        if (pts && pts.length > 0) {
            // Show loading message
            document.getElementById('total-count').textContent = 'Loading...';
            
            createBasemap();
            createMarkers();
            updateCounts();
        } else {
            console.error('No data loaded!');
            document.getElementById('total-count').textContent = 'No data';
        }
        
        // Handle window resize
        window.addEventListener('resize', () => {
            setTimeout(() => {
                mapRect = mapDiv.getBoundingClientRect();
                createBasemap();
                createMarkers();
            }, 100);
        });
    </script>
</body>
</html>
"""
    
    html = html_part1 + js_points + html_part2

    def point_in_poly(x, y, poly):
        # ray casting algorithm for point-in-polygon
        inside = False
        n = len(poly)
        for i in range(n):
            j = (i - 1) % n
            xi, yi = poly[i][1], poly[i][0]  # poly entries are [lat, lon]
            xj, yj = poly[j][1], poly[j][0]
            intersect = ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi)
            if intersect:
                inside = not inside
        return inside

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path in ('/', '/index.html'):
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.send_header('Content-Length', str(len(html.encode('utf-8'))))
                self.end_headers()
                self.wfile.write(html.encode('utf-8'))
            else:
                self.send_response(404)
                self.end_headers()

        def do_POST(self):
            if self.path == '/submit':
                length = int(self.headers.get('content-length', 0))
                body = self.rfile.read(length).decode('utf-8')
                try:
                    payload = json.loads(body)
                except Exception:
                    payload = None
                result = None
                if payload and isinstance(payload.get('polygon'), list):
                    poly = payload['polygon']  # list of [lat, lon]
                    selected = []
                    # server has points available at self.server.points (list of dicts)
                    for idx, p in enumerate(getattr(self.server, 'points', [])):
                        lon = float(p.get('longitude'))
                        lat = float(p.get('latitude'))
                        if point_in_poly(lon, lat, poly):
                            selected.append(idx)
                    result = {'selected': selected}
                    self.server.result = result
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps(result).encode('utf-8'))
                else:
                    # fallback: echo back the payload
                    self.server.result = payload
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({'ok': True}).encode('utf-8'))
                # shutdown after a short delay to allow client receive
                threading.Thread(target=self.server.shutdown, daemon=True).start()
            else:
                self.send_response(404)
                self.end_headers()

    bind_port = port if (port and int(port) > 0) else 0
    with socketserver.ThreadingTCPServer(('127.0.0.1', bind_port), Handler) as httpd:
        port = httpd.server_address[1]
        url = f'http://127.0.0.1:{port}/'
        print(f'Opening browser selector at {url}')
        httpd.result = None
        # expose points to the server handler for selection
        httpd.points = points
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            webbrowser.open(url)
        except Exception:
            print('Open in your browser:', url)

        start = time.time()
        while time.time() - start < timeout:
            if getattr(httpd, 'result', None) is not None:
                res = httpd.result
                try:
                    # If server returned selected indices, write selected CSV if requested
                    if isinstance(res, dict) and res.get('selected') is not None:
                        sel_idx = res['selected']
                        selected_df = df.iloc[sel_idx].reset_index(drop=True)
                        out_path = selected_out or 'selected_catalog.csv'
                        try:
                            selected_df.to_csv(out_path, index=False)
                            print(f'Saved selected catalog to {out_path} ({len(selected_df)} events)')
                        except Exception as e:
                            print('Failed to save selected catalog:', e)
                        return {'selected_count': len(selected_df), 'selected_file': out_path, 'selected_indices': sel_idx}
                    if isinstance(res, dict) and res.get('cancel'):
                        return None
                    # fallback: try to return bbox-like structure if provided
                    return {
                        'lon_min': float(res.get('lon_min', 0.0)), 'lon_max': float(res.get('lon_max', 0.0)),
                        'lat_min': float(res.get('lat_min', 0.0)), 'lat_max': float(res.get('lat_max', 0.0)),
                        'rotation': float(res.get('rotation', 0.0))
                    }
                except Exception:
                    return None
            time.sleep(0.1)
        print('Web selection timed out')
        try:
            httpd.shutdown()
        except Exception:
            pass
        return None


def select_bounds_cli(catalog_df, title="Bounds Selection"):
    print(f"\n=== {title} ===")
    print("Current catalog statistics:")
    print(f"  Events: {len(catalog_df)}")
    print(f"  Longitude: {catalog_df['longitude'].min():.3f} to {catalog_df['longitude'].max():.3f}")
    print(f"  Latitude: {catalog_df['latitude'].min():.3f} to {catalog_df['latitude'].max():.3f}")
    print(f"  Magnitude: {catalog_df['magnitude'].min():.2f} to {catalog_df['magnitude'].max():.2f}")

    print("\nEnter bounding box coordinates:")
    print("(Press Enter to use current values, or 'auto' for automatic selection)")

    default_lon_min = catalog_df['longitude'].min()
    default_lon_max = catalog_df['longitude'].max()
    default_lat_min = catalog_df['latitude'].min()
    default_lat_max = catalog_df['latitude'].max()

    try:
        lon_min_input = input(f"Longitude min [{default_lon_min:.3f}]: ").strip()
        if lon_min_input.lower() == 'auto':
            lon_range = default_lon_max - default_lon_min
            lon_min = default_lon_min + lon_range * 0.2
        elif lon_min_input == '':
            lon_min = default_lon_min
        else:
            lon_min = float(lon_min_input)

        lon_max_input = input(f"Longitude max [{default_lon_max:.3f}]: ").strip()
        if lon_max_input.lower() == 'auto':
            lon_range = default_lon_max - default_lon_min
            lon_max = default_lon_max - lon_range * 0.2
        elif lon_max_input == '':
            lon_max = default_lon_max
        else:
            lon_max = float(lon_max_input)

        lat_min_input = input(f"Latitude min [{default_lat_min:.3f}]: ").strip()
        if lat_min_input.lower() == 'auto':
            lat_range = default_lat_max - default_lat_min
            lat_min = default_lat_min + lat_range * 0.2
        elif lat_min_input == '':
            lat_min = default_lat_min
        else:
            lat_min = float(lat_min_input)

        lat_max_input = input(f"Latitude max [{default_lat_max:.3f}]: ").strip()
        if lat_max_input.lower() == 'auto':
            lat_range = default_lat_max - default_lat_min
            lat_max = default_lat_max - lat_range * 0.2
        elif lat_max_input == '':
            lat_max = default_lat_max
        else:
            lat_max = float(lat_max_input)

        if lon_min >= lon_max or lat_min >= lat_max:
            print("❌ Error: Min values must be less than max values")
            return None

        selected_events = catalog_df[
            (catalog_df['longitude'] >= lon_min) & (catalog_df['longitude'] <= lon_max) &
            (catalog_df['latitude'] >= lat_min) & (catalog_df['latitude'] <= lat_max)
        ]

        print(f"\n✅ Selected {len(selected_events)} events within bounds:")
        print(f"   Longitude: {lon_min:.3f} to {lon_max:.3f}")
        print(f"   Latitude: {lat_min:.3f} to {lat_max:.3f}")

        confirm = input("\nUse these bounds? (y/n): ").strip().lower()
        if confirm in ['y', 'yes', '']:
            return {
                'lon_min': lon_min,
                'lon_max': lon_max,
                'lat_min': lat_min,
                'lat_max': lat_max
            }
        else:
            print("Selection cancelled.")
            return None

    except (ValueError, KeyboardInterrupt) as e:
        print(f"\n❌ Input error: {e}")
        return None


def plot_catalog_basic(catalog_df, title="Earthquake Catalog", figsize=(12, 8)):
    fig, ax = plt.subplots(figsize=figsize)
    scatter = ax.scatter(
        catalog_df['longitude'], catalog_df['latitude'],
        c=catalog_df['magnitude'],
        s=np.maximum(8, catalog_df['magnitude'] * 8),
        cmap='viridis',
        alpha=0.7
    )
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Magnitude')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess earthquake catalog and optionally interactively select bounds',
    )
    parser.add_argument('input_file', help='Input catalog file')
    parser.add_argument('output_file', help='Output processed catalog file')
    parser.add_argument('--min-mag', type=float, default=0.0, help='Minimum magnitude')
    parser.add_argument('--max-depth', type=float, default=20.0, help='Maximum depth (km)')
    parser.add_argument('--plot', action='store_true', help='Create a quick plot of the catalog')
    parser.add_argument('--interactive', action='store_true', help='Use Matplotlib interactive selector')
    parser.add_argument('--interactive-web', action='store_true', help='Use browser-based selector')
    args = parser.parse_args()

    config = {'min_magnitude': args.min_mag, 'max_depth': args.max_depth}
    success = preprocess_catalog(args.input_file, args.output_file, config)
    if not success:
        return

    if args.plot or args.interactive or args.interactive_web:
        try:
            df = pd.read_csv(args.output_file)
            df['time'] = pd.to_datetime(df['time'])
            if args.interactive_web:
                print('Opening browser selector...')
                # write selected catalog exactly to the CLI output filename
                res = plot_catalog_web(df, selected_out=args.output_file)
                print('Result:', res)
            elif args.interactive:
                print('Opening matplotlib interactive selector...')
                res = plot_catalog_interactive(df)
                print('Result:', res)
            else:
                plot_catalog_basic(df)
        except Exception as e:
            print('Plotting failed:', e)
    else:
        print('Preprocessing done. Use --plot or --interactive to inspect the catalog.')


if __name__ == '__main__':
    main()
