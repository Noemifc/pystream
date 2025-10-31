#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HDF5 Image Divider Plugin with Metadata Viewer for PyQtGraph
-------------------------------------------------------------
Opens HDF5 files virtually and displays the division of two image datasets.
Allows real-time shifting of the second image using keyboard arrows.
Includes slider to select which image index to view.
Added metadata viewer tab to display all HDF5 attributes and datasets.

Structure expected:
- /exchange/data (array of projections - first image)
- /exchange/data_white (array of images - second image)

Shows: data / data_white with real-time shift adjustment
Tab 2: Comprehensive metadata viewer
"""

import h5py
import numpy as np
from typing import Optional
from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg
pg.setConfigOptions(imageAxisOrder='row-major')


class Hdf5MetadataReader:
    """
    Metadata reader from meta-cli project
    Reads metadata from HDF5 datasets (not attributes)
    """
    def __init__(self, filePath, excludedSections=['exchange', 'defaults'], readOnOpen=True):
        self.file = h5py.File(filePath, 'r')
        self.metadataDict = {}
        self.excludedSections = excludedSections
        if readOnOpen:
            self.readMetadata()
    
    def readMetadata(self):
        self.file.visititems(self.__readMetadata)
        return self.metadataDict
    
    def getMetadata(self):
        return self.metadataDict
    
    def __readMetadata(self, name, obj):
        if isinstance(obj, h5py.Dataset):
            rootName = name.split('/')[0]
            if rootName not in self.excludedSections:
                try:
                    # This is when the obj shape is (1,) (DESY, APS)
                    if obj[()].shape[0] == 1:
                        value = obj[()][0]
                        if isinstance(value, bytes):
                            value = value.decode("utf-8", errors='ignore')
                        elif (value.dtype.kind == 'S'):
                            value = value.decode(encoding="utf-8")
                        attr = obj.attrs.get('units')
                        if attr != None:
                            attr = attr.decode('UTF-8')
                        self.metadataDict.update({obj.name: [value, attr]})
                except AttributeError:  # This is when the obj is byte so has no attribute 'shape'
                    value = obj[()]
                    if isinstance(value, bytes):
                        value = value.decode("utf-8", errors='ignore')
                    attr = obj.attrs.get('units')
                    if attr != None:
                        attr = attr.decode('UTF-8')
                    self.metadataDict.update({obj.name: [value, attr]})
                except IndexError:  # This is when the obj shape is () (ESRF and DLS) instead of (1,) (DESY, APS)
                    attr = obj.attrs.get('units')
                    if attr != None:
                        if isinstance(attr, str):
                            pass
                        else:
                            attr = attr.decode('UTF-8')
                    value = obj[()]
                    self.metadataDict.update({obj.name: [value, attr]})
    
    def close(self):
        if self.file:
            self.file.close()
            self.file = None


class MetadataExtractor:
    """Extract metadata from HDF5 files using meta-cli approach"""
    
    @staticmethod
    def extract_metadata(h5file):
        """
        Extract metadata from HDF5 file using Hdf5MetadataReader
        Returns a list of tuples: (full_path, value_with_units, dtype)
        """
        metadata = []
        
        # Use the meta-cli reader approach
        # We need to create a temporary reader that uses the already-open file
        temp_reader = Hdf5MetadataReader.__new__(Hdf5MetadataReader)
        temp_reader.file = h5file
        temp_reader.metadataDict = {}
        temp_reader.excludedSections = ['exchange', 'defaults']
        temp_reader.readMetadata()
        
        meta_dict = temp_reader.getMetadata()
        
        # Convert to list format for table display
        for path, (value, units) in meta_dict.items():
            # Format value with units if available
            if units is not None and units != '':
                value_str = f"{value} {units}"
            else:
                value_str = str(value)
            
            # Get dtype
            dtype = type(value).__name__
            if isinstance(value, np.ndarray):
                dtype = f"ndarray({value.dtype})"
            elif isinstance(value, (np.integer, np.floating)):
                dtype = str(value.dtype)
            
            metadata.append((path, value_str, dtype))
        
        return metadata
    
    @staticmethod
    def extract_tree_structure(h5file):
        """
        Extract the tree structure of the HDF5 file
        Returns a list of tuples: (path, type, shape, dtype)
        """
        structure = []
        
        def visit_item(name, obj):
            if isinstance(obj, h5py.Dataset):
                structure.append((name, 'Dataset', obj.shape, obj.dtype))
            elif isinstance(obj, h5py.Group):
                structure.append((name, 'Group', None, None))
        
        h5file.visititems(visit_item)
        return structure


class MetadataViewer(QtWidgets.QWidget):
    """Widget for displaying HDF5 metadata in a table format"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()
    
    def _build_ui(self):
        """Build the metadata viewer interface"""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(10)
        
        # Create tab widget for different views
        self.tab_widget = QtWidgets.QTabWidget()
        
        # Tab 1: Attributes/Metadata
        metadata_widget = QtWidgets.QWidget()
        metadata_layout = QtWidgets.QVBoxLayout(metadata_widget)
        
        # Filter controls
        filter_layout = QtWidgets.QHBoxLayout()
        filter_layout.addWidget(QtWidgets.QLabel("Filter:"))
        
        self.filter_input = QtWidgets.QLineEdit()
        self.filter_input.setPlaceholderText("Type to filter by path or attribute name...")
        self.filter_input.textChanged.connect(self._filter_metadata)
        filter_layout.addWidget(self.filter_input)
        
        metadata_layout.addLayout(filter_layout)
        
        # Metadata table
        self.metadata_table = QtWidgets.QTableWidget()
        self.metadata_table.setColumnCount(3)
        self.metadata_table.setHorizontalHeaderLabels(['Path/Attribute', 'Value', 'Type'])
        self.metadata_table.horizontalHeader().setStretchLastSection(False)
        self.metadata_table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        self.metadata_table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.Interactive)
        self.metadata_table.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeToContents)
        self.metadata_table.setAlternatingRowColors(True)
        self.metadata_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.metadata_table.setSortingEnabled(True)
        metadata_layout.addWidget(self.metadata_table)
        
        # Export button
        export_btn = QtWidgets.QPushButton("Export Metadata to CSV")
        export_btn.clicked.connect(self._export_metadata)
        metadata_layout.addWidget(export_btn)
        
        self.tab_widget.addTab(metadata_widget, "Attributes")
        
        # Tab 2: File Structure
        structure_widget = QtWidgets.QWidget()
        structure_layout = QtWidgets.QVBoxLayout(structure_widget)
        
        # Structure tree
        self.structure_tree = QtWidgets.QTreeWidget()
        self.structure_tree.setHeaderLabels(['Path', 'Type', 'Shape', 'Dtype'])
        self.structure_tree.setAlternatingRowColors(True)
        structure_layout.addWidget(self.structure_tree)
        
        self.tab_widget.addTab(structure_widget, "File Structure")
        
        layout.addWidget(self.tab_widget)
        
        # Status label
        self.status_label = QtWidgets.QLabel("No metadata loaded")
        self.status_label.setStyleSheet("color: #999; padding: 5px;")
        layout.addWidget(self.status_label)
    
    def load_metadata(self, h5file):
        """Load and display metadata from HDF5 file"""
        try:
            # Extract metadata
            metadata = MetadataExtractor.extract_metadata(h5file)
            self._all_metadata = metadata  # Store for filtering
            
            # Populate table
            self._populate_metadata_table(metadata)
            
            # Extract and display structure
            structure = MetadataExtractor.extract_tree_structure(h5file)
            self._populate_structure_tree(h5file, structure)
            
            # Update status
            self.status_label.setText(f"Loaded {len(metadata)} attributes from {len(structure)} objects")
            self.status_label.setStyleSheet("color: #4a4; padding: 5px;")
            
        except Exception as e:
            self.status_label.setText(f"Error loading metadata: {str(e)}")
            self.status_label.setStyleSheet("color: #f44; padding: 5px;")
    
    def _populate_metadata_table(self, metadata):
        """Populate the metadata table with data"""
        self.metadata_table.setSortingEnabled(False)
        self.metadata_table.setRowCount(len(metadata))
        
        for row, (full_path, value, dtype) in enumerate(metadata):
            # Path/Attribute (combined)
            path_item = QtWidgets.QTableWidgetItem(full_path)
            path_item.setFlags(path_item.flags() & ~QtCore.Qt.ItemIsEditable)
            self.metadata_table.setItem(row, 0, path_item)
            
            # Value - format based on type
            if isinstance(value, (int, np.integer)):
                value_str = str(value)
            elif isinstance(value, (float, np.floating)):
                value_str = f"{value:.6g}"
            elif isinstance(value, list):
                # Array/list - show values
                if len(str(value)) > 500:
                    value_str = str(value)[:500] + "..."
                else:
                    value_str = str(value)
            else:
                value_str = str(value)
                if len(value_str) > 500:
                    value_str = value_str[:500] + "..."
            
            value_item = QtWidgets.QTableWidgetItem(value_str)
            value_item.setFlags(value_item.flags() & ~QtCore.Qt.ItemIsEditable)
            value_item.setToolTip(str(value))
            self.metadata_table.setItem(row, 1, value_item)
            
            # Type
            type_item = QtWidgets.QTableWidgetItem(dtype)
            type_item.setFlags(type_item.flags() & ~QtCore.Qt.ItemIsEditable)
            self.metadata_table.setItem(row, 2, type_item)
        
        self.metadata_table.setSortingEnabled(True)
        self.metadata_table.resizeColumnsToContents()
        # Make sure value column is wide enough
        current_width = self.metadata_table.columnWidth(1)
        self.metadata_table.setColumnWidth(1, max(200, current_width))
    
    def _populate_structure_tree(self, h5file, structure):
        """Populate the structure tree with file hierarchy"""
        self.structure_tree.clear()
        
        # Create root item
        root = QtWidgets.QTreeWidgetItem(self.structure_tree)
        root.setText(0, '/')
        root.setText(1, 'Group')
        root.setExpanded(True)
        
        # Build tree
        items_dict = {'/': root}
        
        for path, obj_type, shape, dtype in sorted(structure):
            # Get parent path
            parent_path = '/' + '/'.join(path.split('/')[:-1]) if '/' in path else '/'
            parent_path = parent_path.replace('//', '/')
            
            # Create item
            item = QtWidgets.QTreeWidgetItem()
            item.setText(0, path)
            item.setText(1, obj_type)
            
            if shape is not None:
                item.setText(2, str(shape))
            if dtype is not None:
                item.setText(3, str(dtype))
            
            # Add to parent
            if parent_path in items_dict:
                items_dict[parent_path].addChild(item)
            else:
                root.addChild(item)
            
            items_dict[path] = item
        
        # Expand all
        self.structure_tree.expandAll()
        self.structure_tree.resizeColumnToContents(0)
        self.structure_tree.resizeColumnToContents(1)
    
    def _filter_metadata(self, text):
        """Filter metadata table by search text"""
        if not hasattr(self, '_all_metadata'):
            return
        
        if not text:
            # Show all
            self._populate_metadata_table(self._all_metadata)
        else:
            # Filter by path/attribute name
            filtered = [
                item for item in self._all_metadata
                if text.lower() in item[0].lower()  # item[0] is full_path
            ]
            self._populate_metadata_table(filtered)
    
    def _export_metadata(self):
        """Export metadata to CSV file"""
        if not hasattr(self, '_all_metadata') or not self._all_metadata:
            QtWidgets.QMessageBox.warning(self, "No Data", "No metadata to export")
            return
        
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export Metadata", "", "CSV Files (*.csv)"
        )
        
        if filename:
            try:
                import csv
                with open(filename, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Path/Attribute', 'Value', 'Type'])
                    writer.writerows(self._all_metadata)
                
                QtWidgets.QMessageBox.information(
                    self, "Success", f"Metadata exported to {filename}"
                )
            except Exception as e:
                QtWidgets.QMessageBox.critical(
                    self, "Error", f"Failed to export metadata: {str(e)}"
                )
    
    def clear(self):
        """Clear all metadata"""
        self.metadata_table.setRowCount(0)
        self.structure_tree.clear()
        self.status_label.setText("No metadata loaded")
        self.status_label.setStyleSheet("color: #999; padding: 5px;")


class HDF5ImageDividerDialog(QtWidgets.QDialog):
    """Dialog for viewing HDF5 image division with real-time shifting and metadata"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.hdf5_file = None
        self.data_dataset = None
        self.data_white_dataset = None
        
        # Current state
        self.current_index = 0
        self.shift_x = 0
        self.shift_y = 0
        self.normalization_enabled = True
        
        # Cached images
        self.current_data = None
        self.current_white = None
        self.result_image = None
        
        self.setWindowTitle("HDF5 Image Divider with Metadata Viewer")
        self.setModal(False)
        self.resize(1600, 900)
        
        self._build_ui()
    
    def _build_ui(self):
        """Build the user interface"""
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setSpacing(10)
        
        # Create main tab widget
        self.main_tabs = QtWidgets.QTabWidget()
        
        # Tab 1: Image Viewer
        image_tab = QtWidgets.QWidget()
        self._build_image_tab(image_tab)
        self.main_tabs.addTab(image_tab, "Image Viewer")
        
        # Tab 2: Metadata Viewer
        self.metadata_viewer = MetadataViewer()
        self.main_tabs.addTab(self.metadata_viewer, "Metadata")
        
        main_layout.addWidget(self.main_tabs)
    
    def _build_image_tab(self, parent):
        """Build the image viewer tab"""
        layout = QtWidgets.QHBoxLayout(parent)
        layout.setSpacing(10)
        
        # Left panel - Controls
        left_panel = QtWidgets.QWidget()
        left_panel.setMaximumWidth(350)
        control_layout = QtWidgets.QVBoxLayout(left_panel)
        control_layout.setSpacing(10)
        
        # Title
        title = QtWidgets.QLabel("HDF5 Image Division Tool")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        control_layout.addWidget(title)
        
        # File selection group
        file_group = QtWidgets.QGroupBox("File Selection")
        file_layout = QtWidgets.QVBoxLayout()
        
        self.file_path_label = QtWidgets.QLabel("No file loaded")
        self.file_path_label.setWordWrap(True)
        self.file_path_label.setStyleSheet("color: #999;")
        file_layout.addWidget(self.file_path_label)
        
        load_btn = QtWidgets.QPushButton("Load HDF5 File")
        load_btn.clicked.connect(self._load_file)
        file_layout.addWidget(load_btn)
        
        file_group.setLayout(file_layout)
        control_layout.addWidget(file_group)
        
        # Dataset info group
        info_group = QtWidgets.QGroupBox("Dataset Information")
        info_layout = QtWidgets.QFormLayout()
        
        self.data_shape_label = QtWidgets.QLabel("N/A")
        self.white_shape_label = QtWidgets.QLabel("N/A")
        self.num_images_label = QtWidgets.QLabel("N/A")
        
        info_layout.addRow("Data shape:", self.data_shape_label)
        info_layout.addRow("White shape:", self.white_shape_label)
        info_layout.addRow("Number of images:", self.num_images_label)
        
        info_group.setLayout(info_layout)
        control_layout.addWidget(info_group)
        
        # Image selection group
        selection_group = QtWidgets.QGroupBox("Image Selection")
        selection_layout = QtWidgets.QVBoxLayout()
        
        # Slider for image index
        slider_layout = QtWidgets.QHBoxLayout()
        slider_layout.addWidget(QtWidgets.QLabel("Image Index:"))
        
        self.index_label = QtWidgets.QLabel("0")
        self.index_label.setMinimumWidth(50)
        self.index_label.setStyleSheet("font-weight: bold;")
        slider_layout.addWidget(self.index_label)
        
        selection_layout.addLayout(slider_layout)
        
        self.image_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.image_slider.setMinimum(0)
        self.image_slider.setMaximum(0)
        self.image_slider.setValue(0)
        self.image_slider.setEnabled(False)
        self.image_slider.valueChanged.connect(self._on_slider_changed)
        selection_layout.addWidget(self.image_slider)
        
        selection_group.setLayout(selection_layout)
        control_layout.addWidget(selection_group)
        
        # Normalization control group
        norm_group = QtWidgets.QGroupBox("Normalization")
        norm_layout = QtWidgets.QVBoxLayout()
        
        self.normalization_checkbox = QtWidgets.QCheckBox("Enable Normalization (data / data_white)")
        self.normalization_checkbox.setChecked(True)
        self.normalization_checkbox.stateChanged.connect(self._on_normalization_changed)
        norm_layout.addWidget(self.normalization_checkbox)
        
        self.mode_label = QtWidgets.QLabel("Mode: <b>Division</b>")
        self.mode_label.setStyleSheet("padding: 5px; background-color: #2a2a2a; border-radius: 3px;")
        norm_layout.addWidget(self.mode_label)
        
        norm_group.setLayout(norm_layout)
        control_layout.addWidget(norm_group)
        
        # Contrast/Histogram control group
        contrast_group = QtWidgets.QGroupBox("Contrast Control")
        contrast_layout = QtWidgets.QVBoxLayout()
        
        # Auto-level options
        auto_layout = QtWidgets.QHBoxLayout()
        auto_layout.addWidget(QtWidgets.QLabel("Auto Level:"))
        
        self.auto_level_combo = QtWidgets.QComboBox()
        self.auto_level_combo.addItems([
            "Per Image (default)",
            "Min/Max", 
            "Percentile 1-99%",
            "Percentile 2-98%",
            "Percentile 5-95%",
            "Manual"
        ])
        self.auto_level_combo.currentIndexChanged.connect(self._on_contrast_changed)
        auto_layout.addWidget(self.auto_level_combo)
        contrast_layout.addLayout(auto_layout)
        
        # Manual controls (initially hidden)
        manual_widget = QtWidgets.QWidget()
        manual_layout = QtWidgets.QFormLayout()
        manual_layout.setContentsMargins(0, 0, 0, 0)
        
        self.min_spin = QtWidgets.QDoubleSpinBox()
        self.min_spin.setRange(-1e10, 1e10)
        self.min_spin.setDecimals(4)
        self.min_spin.setValue(0.0)
        self.min_spin.valueChanged.connect(self._on_manual_levels_changed)
        manual_layout.addRow("Min:", self.min_spin)
        
        self.max_spin = QtWidgets.QDoubleSpinBox()
        self.max_spin.setRange(-1e10, 1e10)
        self.max_spin.setDecimals(4)
        self.max_spin.setValue(1.0)
        self.max_spin.valueChanged.connect(self._on_manual_levels_changed)
        manual_layout.addRow("Max:", self.max_spin)
        
        manual_widget.setLayout(manual_layout)
        manual_widget.setVisible(False)
        self.manual_controls = manual_widget
        contrast_layout.addWidget(manual_widget)
        
        # Reset button
        reset_contrast_btn = QtWidgets.QPushButton("Auto Adjust Now")
        reset_contrast_btn.clicked.connect(self._auto_adjust_contrast)
        contrast_layout.addWidget(reset_contrast_btn)
        
        contrast_group.setLayout(contrast_layout)
        control_layout.addWidget(contrast_group)
        
        # Shift control group
        shift_group = QtWidgets.QGroupBox("Shift Control")
        shift_layout = QtWidgets.QFormLayout()
        
        self.shift_x_label = QtWidgets.QLabel("0")
        self.shift_x_label.setStyleSheet("font-weight: bold;")
        shift_layout.addRow("X Shift (pixels):", self.shift_x_label)
        
        self.shift_y_label = QtWidgets.QLabel("0")
        self.shift_y_label.setStyleSheet("font-weight: bold;")
        shift_layout.addRow("Y Shift (pixels):", self.shift_y_label)
        
        # Reset button
        reset_btn = QtWidgets.QPushButton("Reset Shift")
        reset_btn.clicked.connect(self._reset_shift)
        shift_layout.addRow("", reset_btn)
        
        # Instructions
        self.shift_instructions = QtWidgets.QLabel(
            "<b>Keyboard Controls:</b><br>"
            "← → ↑ ↓: Shift image by 1 pixel<br>"
            "Shift + arrows: Shift by 10 pixels<br>"
            "Ctrl + arrows: Shift by 50 pixels"
        )
        self.shift_instructions.setWordWrap(True)
        self.shift_instructions.setStyleSheet("padding: 10px; background-color: #2a2a2a; border-radius: 5px;")
        shift_layout.addRow(self.shift_instructions)
        
        shift_group.setLayout(shift_layout)
        control_layout.addWidget(shift_group)
        
        # Statistics group
        stats_group = QtWidgets.QGroupBox("Image Statistics")
        stats_layout = QtWidgets.QFormLayout()
        
        self.min_val_label = QtWidgets.QLabel("N/A")
        self.max_val_label = QtWidgets.QLabel("N/A")
        self.mean_val_label = QtWidgets.QLabel("N/A")
        self.std_val_label = QtWidgets.QLabel("N/A")
        
        stats_layout.addRow("Min:", self.min_val_label)
        stats_layout.addRow("Max:", self.max_val_label)
        stats_layout.addRow("Mean:", self.mean_val_label)
        stats_layout.addRow("Std Dev:", self.std_val_label)
        
        stats_group.setLayout(stats_layout)
        control_layout.addWidget(stats_group)
        
        control_layout.addStretch()
        
        layout.addWidget(left_panel)
        
        # Right panel - Image display
        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        # PyQtGraph ImageView
        self.image_view = pg.ImageView()
        self.image_view.ui.roiBtn.hide()
        self.image_view.ui.menuBtn.hide()
        right_layout.addWidget(self.image_view)
        
        layout.addWidget(right_panel)
        layout.setStretch(1, 1)
    
    def _load_file(self):
        """Open file dialog and load HDF5 file"""
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open HDF5 File", "", "HDF5 Files (*.h5 *.hdf5);;All Files (*)"
        )
        
        if not filename:
            return
        
        try:
            # Close previous file if exists
            if self.hdf5_file is not None:
                self.hdf5_file.close()
            
            # Open new file
            self.hdf5_file = h5py.File(filename, 'r')
            
            # Try to load expected datasets
            if 'exchange/data' in self.hdf5_file and 'exchange/data_white' in self.hdf5_file:
                self.data_dataset = self.hdf5_file['exchange/data']
                self.data_white_dataset = self.hdf5_file['exchange/data_white']
                
                # Update UI
                self.file_path_label.setText(filename.split('/')[-1])
                self.file_path_label.setStyleSheet("color: white;")
                
                # Update dataset info
                self.data_shape_label.setText(str(self.data_dataset.shape))
                self.white_shape_label.setText(str(self.data_white_dataset.shape))
                
                num_images = self.data_dataset.shape[0]
                self.num_images_label.setText(str(num_images))
                
                # Configure slider
                self.image_slider.setMaximum(num_images - 1)
                self.image_slider.setEnabled(True)
                
                # Load first image
                self._load_and_display_image(0)
                
                # Load metadata into metadata viewer
                self.metadata_viewer.load_metadata(self.hdf5_file)
                
            else:
                QtWidgets.QMessageBox.warning(
                    self, "Invalid File Structure",
                    "File does not contain expected datasets:\n"
                    "- /exchange/data\n"
                    "- /exchange/data_white"
                )
                self.hdf5_file.close()
                self.hdf5_file = None
                
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Error", f"Failed to load file:\n{str(e)}"
            )
            if self.hdf5_file is not None:
                self.hdf5_file.close()
                self.hdf5_file = None
    
    def _load_and_display_image(self, index):
        """Load and display image at given index"""
        if self.data_dataset is None:
            return
        
        try:
            self.current_index = index
            self.index_label.setText(str(index))
            
            # Load data (reading from disk)
            self.current_data = np.array(self.data_dataset[index])
            
            # Use first white image for now (could be made configurable)
            white_index = min(index, self.data_white_dataset.shape[0] - 1)
            self.current_white = np.array(self.data_white_dataset[white_index])
            
            # Update display
            self._update_display()
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Error", f"Failed to load image:\n{str(e)}"
            )
    
    def _update_display(self):
        """Update the image display with current shift and normalization settings"""
        if self.current_data is None:
            return
        
        try:
            if self.normalization_enabled:
                # Apply shift to white image
                shifted_white = self._apply_shift(self.current_white, self.shift_x, self.shift_y)
                
                # Perform division with small epsilon to avoid division by zero
                epsilon = 1e-10
                self.result_image = self.current_data / (shifted_white + epsilon)
                
                # Replace infinities and NaNs
                self.result_image = np.nan_to_num(self.result_image, nan=0.0, posinf=0.0, neginf=0.0)
            else:
                # Show raw data only
                self.result_image = self.current_data.copy()
            
            # Update statistics
            self._update_statistics()
            
            # Apply contrast settings
            self._apply_contrast_settings()
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Error", f"Failed to update display:\n{str(e)}"
            )
    
    def _apply_contrast_settings(self):
        """Apply contrast/level settings to the image"""
        if self.result_image is None:
            return
        
        mode_index = self.auto_level_combo.currentIndex()
        
        if mode_index == 0:  # Per Image (default)
            self.image_view.setImage(self.result_image, autoLevels=True, autoRange=False)
        elif mode_index == 1:  # Min/Max
            vmin, vmax = np.min(self.result_image), np.max(self.result_image)
            self.image_view.setImage(self.result_image, autoLevels=False, autoRange=False, 
                                    levels=(vmin, vmax))
        elif mode_index == 2:  # Percentile 1-99%
            vmin, vmax = np.percentile(self.result_image, [1, 99])
            self.image_view.setImage(self.result_image, autoLevels=False, autoRange=False,
                                    levels=(vmin, vmax))
        elif mode_index == 3:  # Percentile 2-98%
            vmin, vmax = np.percentile(self.result_image, [2, 98])
            self.image_view.setImage(self.result_image, autoLevels=False, autoRange=False,
                                    levels=(vmin, vmax))
        elif mode_index == 4:  # Percentile 5-95%
            vmin, vmax = np.percentile(self.result_image, [5, 95])
            self.image_view.setImage(self.result_image, autoLevels=False, autoRange=False,
                                    levels=(vmin, vmax))
        elif mode_index == 5:  # Manual
            vmin = self.min_spin.value()
            vmax = self.max_spin.value()
            self.image_view.setImage(self.result_image, autoLevels=False, autoRange=False,
                                    levels=(vmin, vmax))
    
    def _update_statistics(self):
        """Update image statistics labels"""
        if self.result_image is None:
            return
        
        self.min_val_label.setText(f"{np.min(self.result_image):.4f}")
        self.max_val_label.setText(f"{np.max(self.result_image):.4f}")
        self.mean_val_label.setText(f"{np.mean(self.result_image):.4f}")
        self.std_val_label.setText(f"{np.std(self.result_image):.4f}")
    
    def _apply_shift(self, image, shift_x, shift_y):
        """Apply x and y shift to an image"""
        if shift_x == 0 and shift_y == 0:
            return image
        
        # Create shifted image
        shifted = np.zeros_like(image)
        
        # Calculate source and destination regions
        src_x_start = max(0, -shift_x)
        src_x_end = image.shape[1] - max(0, shift_x)
        src_y_start = max(0, -shift_y)
        src_y_end = image.shape[0] - max(0, shift_y)
        
        dst_x_start = max(0, shift_x)
        dst_x_end = image.shape[1] - max(0, -shift_x)
        dst_y_start = max(0, shift_y)
        dst_y_end = image.shape[0] - max(0, -shift_y)
        
        # Copy shifted data
        shifted[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
            image[src_y_start:src_y_end, src_x_start:src_x_end]
        
        return shifted
    
    def _on_slider_changed(self, value):
        """Handle slider value change"""
        self._load_and_display_image(value)
    
    def _on_normalization_changed(self, state):
        """Handle normalization checkbox change"""
        self.normalization_enabled = (state == QtCore.Qt.Checked)
        
        # Update mode label
        if self.normalization_enabled:
            self.mode_label.setText("Mode: <b>Division (data / data_white)</b>")
        else:
            self.mode_label.setText("Mode: <b>Raw Data Only</b>")
        
        # Update display
        self._update_display()
    
    def _on_contrast_changed(self, index):
        """Handle contrast mode change"""
        # Show/hide manual controls
        is_manual = (index == 5)
        self.manual_controls.setVisible(is_manual)
        
        # If switching to manual, set initial values from current image
        if is_manual and self.result_image is not None:
            self.min_spin.setValue(float(np.min(self.result_image)))
            self.max_spin.setValue(float(np.max(self.result_image)))
        
        # Update display
        self._update_display()
    
    def _on_manual_levels_changed(self):
        """Handle manual level spinbox changes"""
        if self.auto_level_combo.currentIndex() == 5:  # Only if in manual mode
            self._update_display()
    
    def _auto_adjust_contrast(self):
        """Auto-adjust contrast based on current mode"""
        self._update_display()
    
    def _reset_shift(self):
        """Reset shift to zero"""
        self.shift_x = 0
        self.shift_y = 0
        self._update_shift_labels()
        self._update_display()
    
    def _update_shift_labels(self):
        """Update shift labels"""
        self.shift_x_label.setText(str(self.shift_x))
        self.shift_y_label.setText(str(self.shift_y))
    
    def keyPressEvent(self, event):
        """Handle keyboard events for shifting"""
        if self.current_data is None:
            return
        
        # Only allow shifting when normalization is enabled
        if not self.normalization_enabled:
            super().keyPressEvent(event)
            return
        
        # Determine step size based on modifiers
        step = 1
        if event.modifiers() & QtCore.Qt.ShiftModifier:
            step = 10
        elif event.modifiers() & QtCore.Qt.ControlModifier:
            step = 50
        
        # Handle arrow keys
        if event.key() == QtCore.Qt.Key_Left:
            self.shift_x -= step
            self._update_shift_labels()
            self._update_display()
        elif event.key() == QtCore.Qt.Key_Right:
            self.shift_x += step
            self._update_shift_labels()
            self._update_display()
        elif event.key() == QtCore.Qt.Key_Up:
            self.shift_y -= step
            self._update_shift_labels()
            self._update_display()
        elif event.key() == QtCore.Qt.Key_Down:
            self.shift_y += step
            self._update_shift_labels()
            self._update_display()
        else:
            super().keyPressEvent(event)
    
    def closeEvent(self, event):
        """Clean up when closing"""
        if self.hdf5_file is not None:
            self.hdf5_file.close()
        super().closeEvent(event)


# ==================== Standalone Mode ====================
def main():
    """Run the HDF5 image divider as a standalone application"""
    import sys
    
    # Check if QApplication already exists (for script entry points)
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("HDF5 Image Divider with Metadata")
    
    # Apply dark theme
    app.setStyle('Fusion')
    palette = QtGui.QPalette()
    
    palette.setColor(QtGui.QPalette.Window, QtGui.QColor(53, 53, 53))
    palette.setColor(QtGui.QPalette.WindowText, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.Base, QtGui.QColor(35, 35, 35))
    palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(53, 53, 53))
    palette.setColor(QtGui.QPalette.Text, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.BrightText, QtCore.Qt.red)
    palette.setColor(QtGui.QPalette.Button, QtGui.QColor(53, 53, 53))
    palette.setColor(QtGui.QPalette.ButtonText, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.ToolTipBase, QtGui.QColor(25, 25, 25))
    palette.setColor(QtGui.QPalette.ToolTipText, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(42, 130, 218))
    palette.setColor(QtGui.QPalette.HighlightedText, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, QtGui.QColor(127, 127, 127))
    palette.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.Text, QtGui.QColor(127, 127, 127))
    palette.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.ButtonText, QtGui.QColor(127, 127, 127))
    
    app.setPalette(palette)
    
    app.setStyleSheet("""
        QGroupBox {
            border: 1px solid #555;
            border-radius: 5px;
            margin-top: 10px;
            font-weight: bold;
            padding-top: 10px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px 0 5px;
        }
        QSpinBox, QDoubleSpinBox, QLineEdit {
            background-color: #2a2a2a;
            border: 1px solid #555;
            border-radius: 3px;
            padding: 3px;
            min-height: 20px;
        }
        QSpinBox:focus, QDoubleSpinBox:focus, QLineEdit:focus {
            border: 1px solid #2a82da;
        }
        QPushButton {
            background-color: #454545;
            border: 1px solid #666;
            border-radius: 4px;
            padding: 6px 16px;
            min-width: 80px;
        }
        QPushButton:hover {
            background-color: #505050;
            border: 1px solid #888;
        }
        QPushButton:pressed {
            background-color: #3a3a3a;
        }
        QPushButton:disabled {
            background-color: #353535;
            color: #666;
        }
        QSlider::groove:horizontal {
            border: 1px solid #555;
            height: 8px;
            background: #2a2a2a;
            border-radius: 4px;
        }
        QSlider::handle:horizontal {
            background: #2a82da;
            border: 1px solid #3a95d8;
            width: 18px;
            margin: -5px 0;
            border-radius: 9px;
        }
        QSlider::handle:horizontal:hover {
            background: #3a95d8;
        }
        QTableWidget {
            gridline-color: #555;
            selection-background-color: #2a82da;
        }
        QHeaderView::section {
            background-color: #454545;
            padding: 5px;
            border: 1px solid #555;
            font-weight: bold;
        }
        QTabWidget::pane {
            border: 1px solid #555;
            border-radius: 3px;
        }
        QTabBar::tab {
            background-color: #454545;
            border: 1px solid #555;
            padding: 8px 16px;
            margin-right: 2px;
        }
        QTabBar::tab:selected {
            background-color: #2a82da;
        }
        QTabBar::tab:hover {
            background-color: #505050;
        }
    """)
    
    dialog = HDF5ImageDividerDialog()
    dialog.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()