#!/usr/bin/env python3
"""
Results Tracker - Auto-updates CSV/Excel with experiment results.
Can be synced to Google Sheets.

Usage:
    python results_tracker.py --record ABL-05-1 --accuracy 0.85 --loss 0.42
    python results_tracker.py --summary
    python results_tracker.py --export-excel

Copyright (c) 2024 Rocco Andraeus Sergi
"""

import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import argparse

# =============================================================================
# CONFIGURATION
# =============================================================================

RESULTS_FILE = "results_template.csv"
SUMMARY_FILE = "results_summary.csv"
JSON_BACKUP = "results_data.json"

# =============================================================================
# RESULTS MANAGER
# =============================================================================

class ResultsTracker:
    """Track and update experiment results."""

    def __init__(self, results_dir: str = "."):
        self.results_dir = Path(results_dir)
        self.results_file = self.results_dir / RESULTS_FILE
        self.summary_file = self.results_dir / SUMMARY_FILE
        self.json_file = self.results_dir / JSON_BACKUP
        self.data = self._load_data()

    def _load_data(self) -> List[Dict]:
        """Load existing results from CSV."""
        if not self.results_file.exists():
            return []

        data = []
        with open(self.results_file, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(row)
        return data

    def _save_data(self):
        """Save results to CSV and JSON backup."""
        if not self.data:
            return

        # Save CSV
        fieldnames = self.data[0].keys()
        with open(self.results_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.data)

        # Save JSON backup
        with open(self.json_file, 'w') as f:
            json.dump(self.data, f, indent=2)

    def record_result(
        self,
        experiment_id: str,
        accuracy: float,
        loss: float = None,
        duration_min: float = None,
        cost: float = None,
        accuracies: Dict[str, float] = None,
        notes: str = ""
    ):
        """Record a completed experiment result."""
        # Find the row
        for row in self.data:
            if row.get('Experiment ID') == experiment_id:
                # Update fields
                row['Date'] = datetime.now().strftime("%Y-%m-%d")
                row['End Time'] = datetime.now().strftime("%H:%M")
                row['Overall Accuracy (%)'] = f"{accuracy * 100:.1f}"
                row['Status'] = 'Complete'
                row['Notes'] = notes

                if loss is not None:
                    row['Training Loss'] = f"{loss:.4f}"
                if duration_min is not None:
                    row['Duration (min)'] = f"{duration_min:.1f}"
                if cost is not None:
                    row['GPU Cost ($)'] = f"{cost:.2f}"

                if accuracies:
                    for key, val in accuracies.items():
                        col_name = f"{key} Accuracy (%)"
                        if col_name in row:
                            row[col_name] = f"{val * 100:.1f}"

                self._save_data()
                print(f"✓ Recorded {experiment_id}: {accuracy:.1%} accuracy")
                return True

        print(f"✗ Experiment ID not found: {experiment_id}")
        return False

    def mark_started(self, experiment_id: str):
        """Mark an experiment as started."""
        for row in self.data:
            if row.get('Experiment ID') == experiment_id:
                row['Date'] = datetime.now().strftime("%Y-%m-%d")
                row['Start Time'] = datetime.now().strftime("%H:%M")
                row['Status'] = 'Running'
                self._save_data()
                print(f"▶ Started {experiment_id}")
                return True
        return False

    def mark_failed(self, experiment_id: str, error: str):
        """Mark an experiment as failed."""
        for row in self.data:
            if row.get('Experiment ID') == experiment_id:
                row['Status'] = 'Failed'
                row['Notes'] = error[:100]
                self._save_data()
                print(f"✗ Failed {experiment_id}: {error}")
                return True
        return False

    def get_summary(self) -> Dict:
        """Calculate summary statistics."""
        summary = {
            'total': len(self.data),
            'complete': 0,
            'pending': 0,
            'running': 0,
            'failed': 0,
            'by_experiment': {},
        }

        for row in self.data:
            status = row.get('Status', 'Pending')
            if status == 'Complete':
                summary['complete'] += 1
            elif status == 'Running':
                summary['running'] += 1
            elif status == 'Failed':
                summary['failed'] += 1
            else:
                summary['pending'] += 1

            # Group by experiment type
            exp_type = row.get('Experiment Type', 'Unknown')
            if exp_type not in summary['by_experiment']:
                summary['by_experiment'][exp_type] = {
                    'complete': 0,
                    'total': 0,
                    'accuracies': []
                }
            summary['by_experiment'][exp_type]['total'] += 1

            if status == 'Complete':
                summary['by_experiment'][exp_type]['complete'] += 1
                try:
                    acc = float(row.get('Overall Accuracy (%)', 0)) / 100
                    summary['by_experiment'][exp_type]['accuracies'].append(acc)
                except:
                    pass

        # Calculate means
        for exp_type, stats in summary['by_experiment'].items():
            if stats['accuracies']:
                stats['mean_accuracy'] = sum(stats['accuracies']) / len(stats['accuracies'])
                if len(stats['accuracies']) > 1:
                    mean = stats['mean_accuracy']
                    stats['std_accuracy'] = (sum((a - mean)**2 for a in stats['accuracies']) / len(stats['accuracies'])) ** 0.5
                else:
                    stats['std_accuracy'] = 0

        return summary

    def print_summary(self):
        """Print progress summary."""
        summary = self.get_summary()

        print("\n" + "=" * 50)
        print("  EXPERIMENT PROGRESS")
        print("=" * 50)

        print(f"\nOverall: {summary['complete']}/{summary['total']} complete")
        print(f"  Running: {summary['running']}")
        print(f"  Pending: {summary['pending']}")
        print(f"  Failed: {summary['failed']}")

        print("\nBy Experiment Type:")
        for exp_type, stats in summary['by_experiment'].items():
            pct = stats['complete'] / stats['total'] * 100 if stats['total'] > 0 else 0
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            print(f"  {exp_type:12} {bar} {stats['complete']}/{stats['total']}")

            if 'mean_accuracy' in stats:
                print(f"               Accuracy: {stats['mean_accuracy']:.1%} (±{stats['std_accuracy']:.1%})")

    def get_next_pending(self) -> Optional[str]:
        """Get the next pending experiment ID."""
        for row in self.data:
            if row.get('Status', 'Pending') == 'Pending':
                return row.get('Experiment ID')
        return None

    def export_for_sheets(self) -> str:
        """Export data in Google Sheets-friendly format."""
        # The CSV is already Sheets-compatible
        return str(self.results_file.absolute())


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Track experiment results")
    parser.add_argument('--record', help="Experiment ID to record")
    parser.add_argument('--accuracy', type=float, help="Accuracy (0-1)")
    parser.add_argument('--loss', type=float, help="Training loss")
    parser.add_argument('--duration', type=float, help="Duration in minutes")
    parser.add_argument('--cost', type=float, help="GPU cost in dollars")
    parser.add_argument('--notes', default="", help="Notes")
    parser.add_argument('--start', help="Mark experiment as started")
    parser.add_argument('--fail', help="Mark experiment as failed")
    parser.add_argument('--error', default="Unknown error", help="Error message for failure")
    parser.add_argument('--summary', action='store_true', help="Print summary")
    parser.add_argument('--next', action='store_true', help="Get next pending experiment")
    parser.add_argument('--dir', default=".", help="Results directory")

    args = parser.parse_args()

    tracker = ResultsTracker(args.dir)

    if args.summary:
        tracker.print_summary()

    elif args.next:
        next_exp = tracker.get_next_pending()
        if next_exp:
            print(next_exp)
        else:
            print("All experiments complete!")

    elif args.start:
        tracker.mark_started(args.start)

    elif args.fail:
        tracker.mark_failed(args.fail, args.error)

    elif args.record and args.accuracy is not None:
        tracker.record_result(
            experiment_id=args.record,
            accuracy=args.accuracy,
            loss=args.loss,
            duration_min=args.duration,
            cost=args.cost,
            notes=args.notes
        )
        tracker.print_summary()

    else:
        tracker.print_summary()


if __name__ == "__main__":
    main()
