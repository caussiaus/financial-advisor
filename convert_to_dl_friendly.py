#!/usr/bin/env python3
"""
Convert existing files to DL-friendly storage format.
This script will convert all JSON files and other data files to use the new DL-friendly storage system.
"""

import os
import glob
from pathlib import Path
from src.dl_friendly_storage import convert_existing_json_to_dl_friendly, DLFriendlyStorage

def convert_all_files():
    """Convert all existing files to DL-friendly format"""
    
    # Initialize storage
    storage = DLFriendlyStorage()
    
    # Directories to process
    directories = [
        'data/outputs/analysis_data',
        'data/outputs/ips_output', 
        'data/outputs/reports',
        'data/outputs/client_data',
        'omega_mesh_export',
        'evaluation_results_20250716_175836'
    ]
    
    converted_count = 0
    error_count = 0
    
    for directory in directories:
        if not os.path.exists(directory):
            print(f"⚠️ Directory not found: {directory}")
            continue
            
        print(f"\n📁 Processing directory: {directory}")
        
        # Find all JSON files
        json_files = glob.glob(os.path.join(directory, "*.json"))
        
        for json_file in json_files:
            try:
                # Create backup
                backup_file = json_file + '.backup'
                if not os.path.exists(backup_file):
                    os.rename(json_file, backup_file)
                    print(f"📋 Created backup: {backup_file}")
                
                # Convert to DL-friendly format
                convert_existing_json_to_dl_friendly(backup_file, json_file)
                converted_count += 1
                print(f"✅ Converted: {json_file}")
                
            except Exception as e:
                print(f"❌ Error converting {json_file}: {e}")
                error_count += 1
                
                # Restore backup if conversion failed
                if os.path.exists(backup_file):
                    if os.path.exists(json_file):
                        os.remove(json_file)
                    os.rename(backup_file, json_file)
                    print(f"🔄 Restored backup: {json_file}")
    
    print(f"\n📊 Conversion Summary:")
    print(f"✅ Successfully converted: {converted_count} files")
    print(f"❌ Errors: {error_count} files")
    
    return converted_count, error_count

def convert_mesh_files():
    """Convert mesh-specific files to DL-friendly format"""
    
    mesh_files = [
        'omega_mesh_export/omega_mesh.json',
        'omega_mesh_export/milestones.json'
    ]
    
    print("\n🔄 Converting mesh files...")
    
    for mesh_file in mesh_files:
        if os.path.exists(mesh_file):
            try:
                # Load existing mesh data
                storage = DLFriendlyStorage()
                data = storage.load_from_file(mesh_file, 'json')
                
                # Save in DL-friendly format
                storage.save_to_file(data, mesh_file, 'auto')
                print(f"✅ Converted mesh file: {mesh_file}")
                
            except Exception as e:
                print(f"❌ Error converting mesh file {mesh_file}: {e}")

def convert_analysis_files():
    """Convert analysis data files to DL-friendly format"""
    
    analysis_files = [
        'data/outputs/analysis_data/fake_clients.json',
        'data/outputs/analysis_data/enhanced_pdf_analysis_CLIENT_Case_1_IPS_Individual.json',
        'data/outputs/analysis_data/realistic_life_events_analysis.json',
        'data/outputs/analysis_data/timeline_data_year_4.json',
        'data/outputs/analysis_data/dynamic_portfolio_data.json',
        'data/outputs/analysis_data/stress_dashboard_data.json'
    ]
    
    print("\n📊 Converting analysis files...")
    
    for analysis_file in analysis_files:
        if os.path.exists(analysis_file):
            try:
                # Load existing analysis data
                storage = DLFriendlyStorage()
                data = storage.load_from_file(analysis_file, 'json')
                
                # Save in DL-friendly format
                storage.save_to_file(data, analysis_file, 'auto')
                print(f"✅ Converted analysis file: {analysis_file}")
                
            except Exception as e:
                print(f"❌ Error converting analysis file {analysis_file}: {e}")

def main():
    """Main conversion function"""
    print("🚀 Starting DL-friendly file conversion...")
    print("=" * 50)
    
    # Convert all files
    converted, errors = convert_all_files()
    
    # Convert specific file types
    convert_mesh_files()
    convert_analysis_files()
    
    print("\n" + "=" * 50)
    print("🎉 DL-friendly conversion complete!")
    
    if errors == 0:
        print("✅ All files converted successfully!")
    else:
        print(f"⚠️ {errors} files had conversion errors. Check logs above.")
    
    print("\n📝 Note: Original files have been backed up with .backup extension")
    print("💡 You can restore originals by renaming .backup files back to .json")

if __name__ == "__main__":
    main() 