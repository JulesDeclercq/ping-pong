"""
CSV-based learning module for player AI.

Each player has a CSV file that maps (x, y) paddle positions to (strength, angle_deg, spin_deg).
The learning system:
- 66% uses stored values when position exists in CSV
- 34% random exploration
- Updates CSV on successful shots (ball hit table)
- Diffuses knowledge to nearby positions
"""
import os
import csv
import random
import logging
import numpy as np
from typing import Dict, Tuple, Optional

logger = logging.getLogger("learning")


class LearningPlayer:
    """Manages CSV-based learning for a single player."""
    
    def __init__(self, player_name: str, csv_path: Optional[str] = None, 
                 exploitation_ratio: float = 0.88, diffusion_ratio_1: float = 0.66, 
                 diffusion_ratio_2: float = 0.44):
        """Initialize learning player.
        
        Args:
            player_name: 'paddle1' or 'paddle2'
            csv_path: path to CSV file (defaults to player_name + '.csv')
            exploitation_ratio: 0.0-1.0, probability of using learned values vs random
            diffusion_ratio_1: 0.0-1.0, how similar ±1 neighbors are to center
            diffusion_ratio_2: 0.0-1.0, how similar ±2 neighbors are to center
        """
        self.player_name = player_name
        self.csv_path = csv_path or f"{player_name}.csv"
        self.exploitation_ratio = float(np.clip(exploitation_ratio, 0.0, 1.0))
        self.diffusion_ratio_1 = float(np.clip(diffusion_ratio_1, 0.0, 1.0))
        self.diffusion_ratio_2 = float(np.clip(diffusion_ratio_2, 0.0, 1.0))
        self.data: Dict[Tuple[float, float], Dict[str, float]] = {}
        self.load_csv()
    
    def load_csv(self):
        """Load CSV file into memory. Creates file if it doesn't exist."""
        self.data = {}
        # Ensure CSV file exists (create header) so later writes don't fail
        if not os.path.exists(self.csv_path):
            try:
                with open(self.csv_path, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=['x', 'y', 'strength', 'angle_deg', 'spin_deg', 'count'])
                    writer.writeheader()
            except Exception as e:
                logger.exception("Failed to create CSV file %s: %s", self.csv_path, e)

        if os.path.exists(self.csv_path):
            try:
                with open(self.csv_path, 'r', newline='') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        try:
                            x = round(float(row['x']), 1)
                            y = round(float(row['y']), 1)
                            self.data[(x, y)] = {
                                'strength': float(row['strength']),
                                'angle_deg': float(row['angle_deg']),
                                'spin_deg': float(row['spin_deg']),
                                'count': int(row.get('count', 1))
                            }
                        except (KeyError, ValueError):
                            pass
            except Exception:
                logger.exception("Failed to read CSV %s", self.csv_path)
    
    def save_csv(self):
        """Save current data to CSV file."""
        try:
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['x', 'y', 'strength', 'angle_deg', 'spin_deg', 'count'])
                writer.writeheader()
                for (x, y), vals in sorted(self.data.items()):
                    writer.writerow({
                        'x': round(x, 1),
                        'y': round(y, 1),
                        'strength': round(vals['strength'], 4),
                        'angle_deg': round(vals['angle_deg'], 4),
                        'spin_deg': round(vals['spin_deg'], 4),
                        'count': vals.get('count', 1)
                    })
        except Exception as e:
            logger.exception("Failed to write CSV %s: %s", self.csv_path, e)
        else:
            # Informal console feedback for debugging / smoke tests
            try:
                print(f"[LEARNING] Saved {len(self.data)} entries to '{self.csv_path}'")
            except Exception:
                pass
    
    def get_shot_params(self, x: float, y: float) -> Tuple[float, float, float]:
        """Get (strength, angle_deg, spin_deg) for current paddle position.
        
        If position exists in CSV:
        - 66% chance: use CSV values + small random perturbation
        - 34% chance: use random values (exploration)
        
        If position NOT in CSV:
        - Use random values (pure exploration)
        
        Returns: (strength, angle_deg, spin_deg)
        """
        x = round(float(x), 1)
        y = round(float(y), 1)
        
        if (x, y) in self.data:
            # Position exists in CSV
            if random.random() < self.exploitation_ratio:
                # Use CSV values with small perturbation
                entry = self.data[(x, y)]
                strength = float(np.clip(entry['strength'] + random.uniform(-0.05, 0.05), 0.4, 1.6))
                angle_deg = float(np.clip(entry['angle_deg'] + random.uniform(-2, 2), -33, 33))
                spin_deg = float(np.clip(entry['spin_deg'] + random.uniform(-2, 2), -12, 12))
            else:
                # Random exploration
                strength = float(np.clip(random.uniform(0.4, 1.6), 0.4, 1.6))
                angle_deg = float(np.clip(random.uniform(-33, 33), -33, 33))
                spin_deg = float(np.clip(random.uniform(-12, 12), -12, 12))
        else:
            # Position not in CSV: pure random exploration
            strength = float(np.clip(random.uniform(0.4, 1.6), 0.4, 1.6))
            angle_deg = float(np.clip(random.uniform(-33, 33), -33, 33))
            spin_deg = float(np.clip(random.uniform(-12, 12), -12, 12))
        
        # Return values that will be used for this shot
        self._last_shot = {'x': x, 'y': y, 'strength': strength, 'angle_deg': angle_deg, 'spin_deg': spin_deg}
        return strength, angle_deg, spin_deg
    
    def record_success(self, x: float, y: float, strength: float, angle_deg: float, spin_deg: float):
        """Record a successful shot and update CSV + nearby positions.
        
        Args:
            x, y: position where shot was taken
            strength, angle_deg, spin_deg: parameters that resulted in ball hitting table
        """
        x = round(float(x), 1)
        y = round(float(y), 1)
        strength = float(np.clip(strength, 0.4, 1.6))
        angle_deg = float(np.clip(angle_deg, -33, 33))
        spin_deg = float(np.clip(spin_deg, -12, 12))
        
        # Update or create entry at (x, y)
        if (x, y) in self.data:
            # Position exists: average with new values
            entry = self.data[(x, y)]
            count = entry.get('count', 1)
            entry['strength'] = (entry['strength'] * count + strength) / (count + 1)
            entry['angle_deg'] = (entry['angle_deg'] * count + angle_deg) / (count + 1)
            entry['spin_deg'] = (entry['spin_deg'] * count + spin_deg) / (count + 1)
            entry['count'] = count + 1
        else:
            # New position: create entry
            self.data[(x, y)] = {
                'strength': strength,
                'angle_deg': angle_deg,
                'spin_deg': spin_deg,
                'count': 1
            }
        print(f"[LEARNING] record_success at {(x,y)} -> s={strength:.3f}, ang={angle_deg:.2f}, spin={spin_deg:.2f}")
        
        # Diffuse knowledge to nearby positions
        self._diffuse_knowledge(x, y, strength, angle_deg, spin_deg)
        
        # Save to CSV
        self.save_csv()
    
    def _diffuse_knowledge(self, cx: float, cy: float, strength: float, angle_deg: float, spin_deg: float):
        """Update nearby positions to be similar to the center entry.
        
        - ±1 range: diffusion_ratio_1 similar to center
        - ±2 range: diffusion_ratio_2 similar to center
        """
        center = (round(cx, 1), round(cy, 1))
        center_entry = self.data.get(center)
        if not center_entry:
            return
        
        # Update ±1 range
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue  # Skip center
                nx = round(cx + dx * 0.1, 1)  # 0.1 m step (10 cm grid)
                ny = round(cy + dy * 0.1, 1)
                nkey = (nx, ny)
                
                exploration_ratio_1 = 1.0 - self.diffusion_ratio_1
                if nkey in self.data:
                    entry = self.data[nkey]
                    entry['strength'] = entry['strength'] * exploration_ratio_1 + center_entry['strength'] * self.diffusion_ratio_1
                    entry['angle_deg'] = entry['angle_deg'] * exploration_ratio_1 + center_entry['angle_deg'] * self.diffusion_ratio_1
                    entry['spin_deg'] = entry['spin_deg'] * exploration_ratio_1 + center_entry['spin_deg'] * self.diffusion_ratio_1
                else:
                    # Create new entry with diffusion_ratio_1 similar to center
                    self.data[nkey] = {
                        'strength': center_entry['strength'] * self.diffusion_ratio_1 + random.uniform(0.4, 1.6) * exploration_ratio_1,
                        'angle_deg': center_entry['angle_deg'] * self.diffusion_ratio_1 + random.uniform(-33, 33) * exploration_ratio_1,
                        'spin_deg': center_entry['spin_deg'] * self.diffusion_ratio_1 + random.uniform(-12, 12) * exploration_ratio_1,
                        'count': 1
                    }
        
        # Update ±2 range
        for dx in [-2, -1, 0, 1, 2]:
            for dy in [-2, -1, 0, 1, 2]:
                if abs(dx) <= 1 and abs(dy) <= 1:
                    continue  # Already handled ±1 range
                nx = round(cx + dx * 0.1, 1)  # 0.1 m step (10 cm grid)
                ny = round(cy + dy * 0.1, 1)
                nkey = (nx, ny)
                
                exploration_ratio_2 = 1.0 - self.diffusion_ratio_2
                if nkey in self.data:
                    entry = self.data[nkey]
                    entry['strength'] = entry['strength'] * exploration_ratio_2 + center_entry['strength'] * self.diffusion_ratio_2
                    entry['angle_deg'] = entry['angle_deg'] * exploration_ratio_2 + center_entry['angle_deg'] * self.diffusion_ratio_2
                    entry['spin_deg'] = entry['spin_deg'] * exploration_ratio_2 + center_entry['spin_deg'] * self.diffusion_ratio_2
                else:
                    # Create new entry with diffusion_ratio_2 similar to center
                    self.data[nkey] = {
                        'strength': center_entry['strength'] * self.diffusion_ratio_2 + random.uniform(0.4, 1.6) * exploration_ratio_2,
                        'angle_deg': center_entry['angle_deg'] * self.diffusion_ratio_2 + random.uniform(-33, 33) * exploration_ratio_2,
                        'spin_deg': center_entry['spin_deg'] * self.diffusion_ratio_2 + random.uniform(-12, 12) * exploration_ratio_2,
                        'count': 1
                    }
