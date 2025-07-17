"""
Trial People Manager

This module manages a group of trial people for vector embedding model training:
1. File upload and ingestion for multiple people
2. Surface interpolation across the group
3. Resource management and task scheduling
4. Identification of less dense mesh sections for stress testing
5. High-dimensional topological space visualization
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
import json
import logging
import pickle
from pathlib import Path
import hashlib
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE, MDS
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx

# Import existing components
from src.mesh_vector_database import MeshVectorDatabase, SimilarityMatch
from src.synthetic_lifestyle_engine import SyntheticLifestyleEngine, SyntheticClientData
from src.json_to_vector_converter import ClientVectorProfile, LifeStage, EventCategory


@dataclass
class TrialPerson:
    """Represents a trial person in the system"""
    person_id: str
    name: str
    age: int
    life_stage: LifeStage
    income: float
    net_worth: float
    risk_tolerance: float
    uploaded_files: List[str] = field(default_factory=list)
    mesh_data: Optional[Dict] = None
    vector_embedding: Optional[np.ndarray] = None
    surface_points: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class InterpolatedSurface:
    """Represents an interpolated surface across multiple people"""
    surface_id: str
    surface_type: str  # 'discretionary_spending', 'cash_flow', 'risk_profile'
    grid_points: np.ndarray
    interpolated_values: np.ndarray
    contributing_people: List[str]
    confidence_map: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskSchedule:
    """Represents a scheduled task for the system"""
    task_id: str
    task_type: str  # 'mesh_generation', 'embedding_training', 'surface_interpolation', 'stress_test'
    priority: int  # 1-5, 5 being highest
    estimated_duration: timedelta
    dependencies: List[str] = field(default_factory=list)
    assigned_resources: List[str] = field(default_factory=list)
    status: str = 'pending'  # pending, running, completed, failed
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class TrialPeopleManager:
    """
    Manages trial people for vector embedding model training
    """
    
    def __init__(self, upload_dir: str = "data/inputs/trial_people", 
                 output_dir: str = "data/outputs/trial_analysis"):
        self.upload_dir = Path(upload_dir)
        self.output_dir = Path(output_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Trial people storage
        self.trial_people: Dict[str, TrialPerson] = {}
        self.vector_db = MeshVectorDatabase(embedding_dim=128, similarity_threshold=0.7)
        self.synthetic_engine = SyntheticLifestyleEngine(use_gpu=False)
        
        # Surface interpolation
        self.interpolated_surfaces: Dict[str, InterpolatedSurface] = {}
        
        # Task scheduling
        self.task_queue: List[TaskSchedule] = []
        self.completed_tasks: List[TaskSchedule] = []
        self.resources = {
            'cpu_cores': 4,
            'memory_gb': 8,
            'gpu_available': False,
            'mesh_processing_slots': 2
        }
        
        # Setup logging
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the trial people manager"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def get_upload_instructions(self) -> str:
        """Get instructions for file uploads"""
        instructions = f"""
📁 UPLOAD INSTRUCTIONS FOR TRIAL PEOPLE

Please upload files for each trial person to: {self.upload_dir}

For each person, create a folder with their name and include:

1. PERSONAL_INFO.json - Basic information:
   {{
     "name": "John Doe",
     "age": 35,
     "income": 75000,
     "net_worth": 150000,
     "risk_tolerance": 0.6,
     "life_stage": "mid_career"
   }}

2. LIFESTYLE_EVENTS.json - Expected life events:
   {{
     "events": [
       {{
         "event_type": "career_change",
         "expected_age": 38,
         "estimated_cost": 5000,
         "probability": 0.7
       }}
     ]
   }}

3. FINANCIAL_PROFILE.json - Current financial situation:
   {{
     "monthly_income": 6250,
     "monthly_expenses": 4000,
     "savings_rate": 0.2,
     "debt_to_income_ratio": 0.3,
     "investment_portfolio": {{
       "stocks": 0.6,
       "bonds": 0.3,
       "cash": 0.1
     }}
   }}

4. GOALS.json - Financial goals:
   {{
     "short_term_goals": ["emergency_fund", "debt_payoff"],
     "medium_term_goals": ["house_down_payment", "education_fund"],
     "long_term_goals": ["retirement_savings", "estate_planning"]
   }}

Example folder structure:
{self.upload_dir}/
├── john_doe/
│   ├── PERSONAL_INFO.json
│   ├── LIFESTYLE_EVENTS.json
│   ├── FINANCIAL_PROFILE.json
│   └── GOALS.json
├── jane_smith/
│   ├── PERSONAL_INFO.json
│   ├── LIFESTYLE_EVENTS.json
│   ├── FINANCIAL_PROFILE.json
│   └── GOALS.json
└── ...

The system will automatically process these files and generate:
- Vector embeddings for each person
- Mesh network compositions
- Interpolated surfaces across the group
- Task scheduling for analysis
- High-dimensional topology visualizations
"""
        return instructions
    
    def scan_upload_directory(self) -> List[str]:
        """Scan upload directory for trial people folders"""
        people_folders = []
        
        if self.upload_dir.exists():
            for folder in self.upload_dir.iterdir():
                if folder.is_dir():
                    # Check if folder contains required files
                    required_files = ['PERSONAL_INFO.json', 'LIFESTYLE_EVENTS.json', 
                                   'FINANCIAL_PROFILE.json', 'GOALS.json']
                    
                    has_all_files = all((folder / file).exists() for file in required_files)
                    if has_all_files:
                        people_folders.append(folder.name)
        
        return people_folders
    
    def ingest_trial_person(self, person_folder: str) -> TrialPerson:
        """Ingest a trial person from their folder"""
        folder_path = self.upload_dir / person_folder
        
        # Load personal info
        with open(folder_path / 'PERSONAL_INFO.json', 'r') as f:
            personal_info = json.load(f)
        
        # Load lifestyle events
        with open(folder_path / 'LIFESTYLE_EVENTS.json', 'r') as f:
            lifestyle_events = json.load(f)
        
        # Load financial profile
        with open(folder_path / 'FINANCIAL_PROFILE.json', 'r') as f:
            financial_profile = json.load(f)
        
        # Load goals
        with open(folder_path / 'GOALS.json', 'r') as f:
            goals = json.load(f)
        
        # Create trial person
        person = TrialPerson(
            person_id=person_folder,
            name=personal_info['name'],
            age=personal_info['age'],
            life_stage=LifeStage(personal_info['life_stage']),
            income=personal_info['income'],
            net_worth=personal_info['net_worth'],
            risk_tolerance=personal_info['risk_tolerance'],
            uploaded_files=[str(f) for f in folder_path.glob('*.json')],
            metadata={
                'lifestyle_events': lifestyle_events,
                'financial_profile': financial_profile,
                'goals': goals
            }
        )
        
        self.trial_people[person.person_id] = person
        self.logger.info(f"Ingested trial person: {person.name} ({person.person_id})")
        
        return person
    
    def process_trial_person_with_mesh(self, person: TrialPerson) -> TrialPerson:
        """Process a trial person with the mesh engine"""
        self.logger.info(f"Processing {person.name} with mesh engine...")
        
        # Convert to synthetic client data
        client_data = self._convert_trial_person_to_client_data(person)
        
        # Process with mesh engine
        client_data = self.synthetic_engine.process_with_mesh_engine(
            client_data,
            num_scenarios=500,  # More scenarios for trial people
            time_horizon_years=5  # Longer horizon for trial people
        )
        
        # Extract mesh data
        person.mesh_data = client_data.mesh_data
        person.vector_embedding = self.vector_db._generate_embedding_vector(
            self.vector_db._generate_mesh_composition_features(client_data)
        )
        
        # Add to vector database
        self.vector_db.add_client(client_data)
        
        self.logger.info(f"Completed mesh processing for {person.name}")
        return person
    
    def _convert_trial_person_to_client_data(self, person: TrialPerson) -> SyntheticClientData:
        """Convert trial person to synthetic client data"""
        # Create basic profile
        profile_data = {
            'name': person.name,
            'age': person.age,
            'base_income': person.income,
            'risk_tolerance': person.risk_tolerance
        }
        
        # Create events from lifestyle events
        events_data = []
        for event in person.metadata['lifestyle_events']['events']:
            events_data.append({
                'event_type': event['event_type'],
                'expected_age': event['expected_age'],
                'estimated_cost': event['estimated_cost'],
                'probability': event['probability']
            })
        
        # Create financial metrics
        financial_data = person.metadata['financial_profile']
        financial_metrics = {
            'net_worth': person.net_worth,
            'monthly_income': financial_data['monthly_income'],
            'monthly_expenses': financial_data['monthly_expenses'],
            'savings_rate': financial_data['savings_rate'],
            'debt_to_income_ratio': financial_data['debt_to_income_ratio']
        }
        
        # Generate synthetic client data
        client_data = self.synthetic_engine.generate_synthetic_client(
            target_age=profile_data['age']
        )
        
        return client_data
    
    def interpolate_surfaces_across_group(self) -> Dict[str, InterpolatedSurface]:
        """Interpolate surfaces across the trial people group"""
        self.logger.info("Interpolating surfaces across trial people group...")
        
        if len(self.trial_people) < 2:
            self.logger.warning("Need at least 2 people for surface interpolation")
            return {}
        
        # Get all people with mesh data
        people_with_mesh = [p for p in self.trial_people.values() if p.mesh_data is not None]
        
        if len(people_with_mesh) < 2:
            self.logger.warning("Need at least 2 people with mesh data for interpolation")
            return {}
        
        surfaces = {}
        
        # Interpolate discretionary spending surface
        discretionary_surface = self._interpolate_discretionary_spending(people_with_mesh)
        if discretionary_surface:
            surfaces['discretionary_spending'] = discretionary_surface
        
        # Interpolate cash flow surface
        cash_flow_surface = self._interpolate_cash_flow(people_with_mesh)
        if cash_flow_surface:
            surfaces['cash_flow'] = cash_flow_surface
        
        # Interpolate risk profile surface
        risk_surface = self._interpolate_risk_profile(people_with_mesh)
        if risk_surface:
            surfaces['risk_profile'] = risk_surface
        
        self.interpolated_surfaces.update(surfaces)
        self.logger.info(f"Generated {len(surfaces)} interpolated surfaces")
        
        return surfaces
    
    def _interpolate_discretionary_spending(self, people: List[TrialPerson]) -> Optional[InterpolatedSurface]:
        """Interpolate discretionary spending surface across people"""
        try:
            # Extract discretionary spending data
            spending_data = []
            for person in people:
                if person.mesh_data and 'discretionary_spending' in person.mesh_data:
                    spending_data.append({
                        'person_id': person.person_id,
                        'age': person.age,
                        'income': person.income,
                        'spending_surface': person.mesh_data['discretionary_spending']
                    })
            
            if len(spending_data) < 2:
                return None
            
            # Create interpolation grid
            ages = [d['age'] for d in spending_data]
            incomes = [d['income'] for d in spending_data]
            
            age_range = (min(ages), max(ages))
            income_range = (min(incomes), max(incomes))
            
            # Create grid points
            age_grid = np.linspace(age_range[0], age_range[1], 20)
            income_grid = np.linspace(income_range[0], income_range[1], 20)
            age_mesh, income_mesh = np.meshgrid(age_grid, income_grid)
            
            # Interpolate values
            interpolated_values = np.zeros_like(age_mesh)
            confidence_map = np.zeros_like(age_mesh)
            
            for i in range(age_mesh.shape[0]):
                for j in range(age_mesh.shape[1]):
                    # Simple inverse distance weighting interpolation
                    total_weight = 0
                    weighted_sum = 0
                    
                    for data in spending_data:
                        distance = np.sqrt((age_mesh[i, j] - data['age'])**2 + 
                                        (income_mesh[i, j] - data['income'])**2)
                        if distance > 0:
                            weight = 1 / distance
                            weighted_sum += weight * np.mean(data['spending_surface'])
                            total_weight += weight
                    
                    if total_weight > 0:
                        interpolated_values[i, j] = weighted_sum / total_weight
                        confidence_map[i, j] = min(1.0, total_weight / len(spending_data))
            
            surface = InterpolatedSurface(
                surface_id=f"discretionary_spending_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                surface_type='discretionary_spending',
                grid_points=np.stack([age_mesh.flatten(), income_mesh.flatten()], axis=1),
                interpolated_values=interpolated_values.flatten(),
                contributing_people=[p.person_id for p in people],
                confidence_map=confidence_map.flatten()
            )
            
            return surface
            
        except Exception as e:
            self.logger.error(f"Error interpolating discretionary spending: {e}")
            return None
    
    def _interpolate_cash_flow(self, people: List[TrialPerson]) -> Optional[InterpolatedSurface]:
        """Interpolate cash flow surface across people"""
        try:
            # Extract cash flow data
            cash_flow_data = []
            for person in people:
                if person.mesh_data and 'cash_flow_vector' in person.mesh_data:
                    cash_flow_data.append({
                        'person_id': person.person_id,
                        'age': person.age,
                        'income': person.income,
                        'cash_flow': person.mesh_data['cash_flow_vector']
                    })
            
            if len(cash_flow_data) < 2:
                return None
            
            # Create interpolation grid (similar to discretionary spending)
            ages = [d['age'] for d in cash_flow_data]
            incomes = [d['income'] for d in cash_flow_data]
            
            age_range = (min(ages), max(ages))
            income_range = (min(incomes), max(incomes))
            
            age_grid = np.linspace(age_range[0], age_range[1], 20)
            income_grid = np.linspace(income_range[0], income_range[1], 20)
            age_mesh, income_mesh = np.meshgrid(age_grid, income_grid)
            
            interpolated_values = np.zeros_like(age_mesh)
            confidence_map = np.zeros_like(age_mesh)
            
            for i in range(age_mesh.shape[0]):
                for j in range(age_mesh.shape[1]):
                    total_weight = 0
                    weighted_sum = 0
                    
                    for data in cash_flow_data:
                        distance = np.sqrt((age_mesh[i, j] - data['age'])**2 + 
                                        (income_mesh[i, j] - data['income'])**2)
                        if distance > 0:
                            weight = 1 / distance
                            weighted_sum += weight * np.mean(data['cash_flow'])
                            total_weight += weight
                    
                    if total_weight > 0:
                        interpolated_values[i, j] = weighted_sum / total_weight
                        confidence_map[i, j] = min(1.0, total_weight / len(cash_flow_data))
            
            surface = InterpolatedSurface(
                surface_id=f"cash_flow_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                surface_type='cash_flow',
                grid_points=np.stack([age_mesh.flatten(), income_mesh.flatten()], axis=1),
                interpolated_values=interpolated_values.flatten(),
                contributing_people=[p.person_id for p in people],
                confidence_map=confidence_map.flatten()
            )
            
            return surface
            
        except Exception as e:
            self.logger.error(f"Error interpolating cash flow: {e}")
            return None
    
    def _interpolate_risk_profile(self, people: List[TrialPerson]) -> Optional[InterpolatedSurface]:
        """Interpolate risk profile surface across people"""
        try:
            # Extract risk profile data
            risk_data = []
            for person in people:
                if person.mesh_data and 'risk_analysis' in person.mesh_data:
                    risk_analysis = person.mesh_data['risk_analysis']
                    risk_data.append({
                        'person_id': person.person_id,
                        'age': person.age,
                        'risk_tolerance': person.risk_tolerance,
                        'var_95': np.mean(risk_analysis.get('var_95_timeline', [0])),
                        'max_drawdown': np.max(risk_analysis.get('max_drawdown_by_scenario', [0]))
                    })
            
            if len(risk_data) < 2:
                return None
            
            # Create interpolation grid
            ages = [d['age'] for d in risk_data]
            risk_tolerances = [d['risk_tolerance'] for d in risk_data]
            
            age_range = (min(ages), max(ages))
            risk_range = (min(risk_tolerances), max(risk_tolerances))
            
            age_grid = np.linspace(age_range[0], age_range[1], 20)
            risk_grid = np.linspace(risk_range[0], risk_range[1], 20)
            age_mesh, risk_mesh = np.meshgrid(age_grid, risk_grid)
            
            interpolated_values = np.zeros_like(age_mesh)
            confidence_map = np.zeros_like(age_mesh)
            
            for i in range(age_mesh.shape[0]):
                for j in range(age_mesh.shape[1]):
                    total_weight = 0
                    weighted_sum = 0
                    
                    for data in risk_data:
                        distance = np.sqrt((age_mesh[i, j] - data['age'])**2 + 
                                        (risk_mesh[i, j] - data['risk_tolerance'])**2)
                        if distance > 0:
                            weight = 1 / distance
                            weighted_sum += weight * data['var_95']
                            total_weight += weight
                    
                    if total_weight > 0:
                        interpolated_values[i, j] = weighted_sum / total_weight
                        confidence_map[i, j] = min(1.0, total_weight / len(risk_data))
            
            surface = InterpolatedSurface(
                surface_id=f"risk_profile_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                surface_type='risk_profile',
                grid_points=np.stack([age_mesh.flatten(), risk_mesh.flatten()], axis=1),
                interpolated_values=interpolated_values.flatten(),
                contributing_people=[p.person_id for p in people],
                confidence_map=confidence_map.flatten()
            )
            
            return surface
            
        except Exception as e:
            self.logger.error(f"Error interpolating risk profile: {e}")
            return None
    
    def schedule_tasks(self) -> List[TaskSchedule]:
        """Schedule tasks for the trial people analysis"""
        self.logger.info("Scheduling tasks for trial people analysis...")
        
        tasks = []
        
        # Task 1: Process all trial people with mesh engine
        mesh_task = TaskSchedule(
            task_id=f"mesh_processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            task_type='mesh_generation',
            priority=5,
            estimated_duration=timedelta(minutes=30),
            dependencies=[],
            assigned_resources=['mesh_processing_slots']
        )
        tasks.append(mesh_task)
        
        # Task 2: Surface interpolation
        interpolation_task = TaskSchedule(
            task_id=f"surface_interpolation_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            task_type='surface_interpolation',
            priority=4,
            estimated_duration=timedelta(minutes=15),
            dependencies=[mesh_task.task_id],
            assigned_resources=['cpu_cores']
        )
        tasks.append(interpolation_task)
        
        # Task 3: Vector embedding training
        embedding_task = TaskSchedule(
            task_id=f"embedding_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            task_type='embedding_training',
            priority=3,
            estimated_duration=timedelta(minutes=20),
            dependencies=[mesh_task.task_id],
            assigned_resources=['cpu_cores']
        )
        tasks.append(embedding_task)
        
        # Task 4: Stress testing
        stress_task = TaskSchedule(
            task_id=f"stress_testing_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            task_type='stress_test',
            priority=2,
            estimated_duration=timedelta(minutes=25),
            dependencies=[interpolation_task.task_id, embedding_task.task_id],
            assigned_resources=['cpu_cores']
        )
        tasks.append(stress_task)
        
        # Task 5: Visualization generation
        viz_task = TaskSchedule(
            task_id=f"visualization_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            task_type='visualization',
            priority=1,
            estimated_duration=timedelta(minutes=10),
            dependencies=[stress_task.task_id],
            assigned_resources=['cpu_cores']
        )
        tasks.append(viz_task)
        
        self.task_queue.extend(tasks)
        self.logger.info(f"Scheduled {len(tasks)} tasks")
        
        return tasks
    
    def identify_less_dense_sections(self) -> Dict[str, Any]:
        """Identify less dense sections of the mesh for stress testing"""
        self.logger.info("Identifying less dense mesh sections...")
        
        if len(self.trial_people) < 3:
            self.logger.warning("Need at least 3 people to identify less dense sections")
            return {}
        
        # Get embeddings for all people
        embeddings = []
        people_ids = []
        
        for person in self.trial_people.values():
            if person.vector_embedding is not None:
                embeddings.append(person.vector_embedding)
                people_ids.append(person.person_id)
        
        if len(embeddings) < 3:
            return {}
        
        embeddings = np.array(embeddings)
        
        # Use DBSCAN to identify clusters and outliers
        clustering = DBSCAN(eps=0.3, min_samples=2).fit(embeddings)
        labels = clustering.labels_
        
        # Identify outliers (less dense sections)
        outliers = labels == -1
        outlier_indices = np.where(outliers)[0]
        
        # Calculate density scores
        density_scores = []
        for i, embedding in enumerate(embeddings):
            distances = np.linalg.norm(embeddings - embedding, axis=1)
            # Remove self-distance
            distances = distances[distances > 0]
            if len(distances) > 0:
                # Lower average distance = higher density
                density_score = 1 / (1 + np.mean(distances))
                density_scores.append(density_score)
            else:
                density_scores.append(0.0)
        
        # Identify low-density regions
        density_threshold = np.percentile(density_scores, 25)  # Bottom 25%
        low_density_indices = [i for i, score in enumerate(density_scores) if score < density_threshold]
        
        analysis = {
            'total_people': len(embeddings),
            'clusters_found': len(set(labels)) - (1 if -1 in labels else 0),
            'outliers_found': len(outlier_indices),
            'low_density_people': [people_ids[i] for i in low_density_indices],
            'outlier_people': [people_ids[i] for i in outlier_indices],
            'density_scores': {people_ids[i]: density_scores[i] for i in range(len(people_ids))},
            'clustering_labels': {people_ids[i]: int(labels[i]) for i in range(len(people_ids))}
        }
        
        self.logger.info(f"Found {len(outlier_indices)} outliers and {len(low_density_indices)} low-density people")
        
        return analysis
    
    def visualize_high_dimensional_topology(self) -> Dict[str, str]:
        """Visualize the high-dimensional topological space"""
        self.logger.info("Creating high-dimensional topology visualizations...")
        
        if len(self.trial_people) < 2:
            self.logger.warning("Need at least 2 people for topology visualization")
            return {}
        
        # Get embeddings
        embeddings = []
        people_info = []
        
        for person in self.trial_people.values():
            if person.vector_embedding is not None:
                embeddings.append(person.vector_embedding)
                people_info.append({
                    'id': person.person_id,
                    'name': person.name,
                    'age': person.age,
                    'life_stage': person.life_stage.value
                })
        
        if len(embeddings) < 2:
            return {}
        
        embeddings = np.array(embeddings)
        
        # Create visualizations
        viz_files = {}
        
        # 1. PCA visualization
        pca = PCA(n_components=2)
        pca_embeddings = pca.fit_transform(embeddings)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        scatter = ax.scatter(pca_embeddings[:, 0], pca_embeddings[:, 1], 
                           c=[info['age'] for info in people_info], 
                           cmap='viridis', s=100, alpha=0.7)
        
        for i, info in enumerate(people_info):
            ax.annotate(info['name'], (pca_embeddings[i, 0], pca_embeddings[i, 1]),
                       xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        ax.set_title('High-Dimensional Topology: PCA Projection')
        plt.colorbar(scatter, label='Age')
        plt.tight_layout()
        
        pca_file = self.output_dir / 'pca_topology.png'
        plt.savefig(pca_file, dpi=300, bbox_inches='tight')
        viz_files['pca_topology'] = str(pca_file)
        plt.close()
        
        # 2. t-SNE visualization
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(3, len(embeddings)-1))
        tsne_embeddings = tsne.fit_transform(embeddings)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        scatter = ax.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1],
                           c=[info['age'] for info in people_info],
                           cmap='viridis', s=100, alpha=0.7)
        
        for i, info in enumerate(people_info):
            ax.annotate(info['name'], (tsne_embeddings[i, 0], tsne_embeddings[i, 1]),
                       xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax.set_xlabel('t-SNE Dimension 1')
        ax.set_ylabel('t-SNE Dimension 2')
        ax.set_title('High-Dimensional Topology: t-SNE Projection')
        plt.colorbar(scatter, label='Age')
        plt.tight_layout()
        
        tsne_file = self.output_dir / 'tsne_topology.png'
        plt.savefig(tsne_file, dpi=300, bbox_inches='tight')
        viz_files['tsne_topology'] = str(tsne_file)
        plt.close()
        
        # 3. Interactive 3D visualization with Plotly
        if len(embeddings) >= 3:
            # Use first 3 PCA components for 3D visualization
            pca_3d = PCA(n_components=3)
            pca_3d_embeddings = pca_3d.fit_transform(embeddings)
            
            fig = go.Figure(data=[go.Scatter3d(
                x=pca_3d_embeddings[:, 0],
                y=pca_3d_embeddings[:, 1],
                z=pca_3d_embeddings[:, 2],
                mode='markers+text',
                marker=dict(
                    size=10,
                    color=[info['age'] for info in people_info],
                    colorscale='Viridis',
                    opacity=0.8
                ),
                text=[info['name'] for info in people_info],
                textposition="middle center"
            )])
            
            fig.update_layout(
                title='High-Dimensional Topology: 3D PCA Projection',
                scene=dict(
                    xaxis_title=f'PC1 ({pca_3d.explained_variance_ratio_[0]:.2%} variance)',
                    yaxis_title=f'PC2 ({pca_3d.explained_variance_ratio_[1]:.2%} variance)',
                    zaxis_title=f'PC3 ({pca_3d.explained_variance_ratio_[2]:.2%} variance)'
                ),
                width=1000,
                height=800
            )
            
            plotly_file = self.output_dir / '3d_topology.html'
            fig.write_html(str(plotly_file))
            viz_files['3d_topology'] = str(plotly_file)
        
        # 4. Network graph visualization
        # Create similarity network
        similarity_matrix = np.zeros((len(embeddings), len(embeddings)))
        for i in range(len(embeddings)):
            for j in range(len(embeddings)):
                if i != j:
                    similarity = np.dot(embeddings[i], embeddings[j])
                    similarity_matrix[i, j] = similarity
        
        # Create network graph
        G = nx.Graph()
        for i, info in enumerate(people_info):
            G.add_node(info['id'], name=info['name'], age=info['age'])
        
        # Add edges based on similarity
        for i in range(len(embeddings)):
            for j in range(i+1, len(embeddings)):
                similarity = similarity_matrix[i, j]
                if similarity > 0.5:  # Threshold for edge creation
                    G.add_edge(people_info[i]['id'], people_info[j]['id'], 
                              weight=similarity)
        
        # Create network visualization
        plt.figure(figsize=(12, 10))
        pos = nx.spring_layout(G, k=3, iterations=50)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, 
                              node_color=[info['age'] for info in people_info],
                              node_size=1000, 
                              cmap=plt.cm.viridis,
                              alpha=0.8)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, alpha=0.3)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, 
                               {info['id']: info['name'] for info in people_info},
                               font_size=8)
        
        plt.title('High-Dimensional Topology: Similarity Network')
        
        # Create a proper mappable for the colorbar
        norm = plt.Normalize(min([info['age'] for info in people_info]), 
                           max([info['age'] for info in people_info]))
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, label='Age', ax=plt.gca())
        plt.tight_layout()
        
        network_file = self.output_dir / 'network_topology.png'
        plt.savefig(network_file, dpi=300, bbox_inches='tight')
        viz_files['network_topology'] = str(network_file)
        plt.close()
        
        self.logger.info(f"Created {len(viz_files)} topology visualizations")
        return viz_files
    
    def save_analysis_results(self) -> str:
        """Save all analysis results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.output_dir / f"trial_analysis_results_{timestamp}.json"
        
        results = {
            'timestamp': timestamp,
            'trial_people': {
                person_id: {
                    'name': person.name,
                    'age': person.age,
                    'life_stage': person.life_stage.value,
                    'income': person.income,
                    'net_worth': person.net_worth,
                    'risk_tolerance': person.risk_tolerance,
                    'has_mesh_data': person.mesh_data is not None,
                    'has_embedding': person.vector_embedding is not None
                }
                for person_id, person in self.trial_people.items()
            },
            'interpolated_surfaces': {
                surface_id: {
                    'surface_type': surface.surface_type,
                    'contributing_people': surface.contributing_people,
                    'grid_shape': surface.grid_points.shape,
                    'value_range': [float(np.min(surface.interpolated_values)), 
                                  float(np.max(surface.interpolated_values))]
                }
                for surface_id, surface in self.interpolated_surfaces.items()
            },
            'task_schedule': {
                'queued_tasks': len(self.task_queue),
                'completed_tasks': len(self.completed_tasks)
            },
            'vector_database_stats': {
                'total_embeddings': len(self.vector_db.embeddings),
                'embedding_dimension': self.vector_db.embedding_dim
            }
        }
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.logger.info(f"Saved analysis results to {results_file}")
        return str(results_file)


def create_trial_people_demo():
    """Create a demo with 5 trial people"""
    print("🚀 Trial People Manager Demo")
    print("=" * 50)
    
    # Create trial people manager
    manager = TrialPeopleManager()
    
    # Show upload instructions
    print("\n📁 Upload Instructions:")
    print(manager.get_upload_instructions())
    
    # Scan for uploaded people
    print("\n🔍 Scanning for uploaded trial people...")
    people_folders = manager.scan_upload_directory()
    
    if people_folders:
        print(f"Found {len(people_folders)} trial people: {people_folders}")
        
        # Ingest trial people
        print("\n📥 Ingesting trial people...")
        for folder in people_folders:
            person = manager.ingest_trial_person(folder)
            print(f"   ✅ Ingested {person.name} ({person.person_id})")
        
        # Process with mesh engine
        print("\n🌐 Processing with mesh engine...")
        for person in manager.trial_people.values():
            person = manager.process_trial_person_with_mesh(person)
            print(f"   ✅ Processed {person.name} with mesh engine")
        
        # Interpolate surfaces
        print("\n📊 Interpolating surfaces...")
        surfaces = manager.interpolate_surfaces_across_group()
        print(f"   ✅ Generated {len(surfaces)} interpolated surfaces")
        
        # Schedule tasks
        print("\n📅 Scheduling tasks...")
        tasks = manager.schedule_tasks()
        print(f"   ✅ Scheduled {len(tasks)} tasks")
        
        # Identify less dense sections
        print("\n🎯 Identifying less dense sections...")
        density_analysis = manager.identify_less_dense_sections()
        if density_analysis:
            print(f"   ✅ Found {density_analysis['outliers_found']} outliers")
            print(f"   ✅ Found {len(density_analysis['low_density_people'])} low-density people")
        
        # Create visualizations
        print("\n📈 Creating topology visualizations...")
        viz_files = manager.visualize_high_dimensional_topology()
        print(f"   ✅ Created {len(viz_files)} visualizations")
        
        # Save results
        print("\n💾 Saving analysis results...")
        results_file = manager.save_analysis_results()
        print(f"   ✅ Saved results to {results_file}")
        
        return manager, people_folders, surfaces, density_analysis, viz_files
    else:
        print("❌ No trial people found. Please upload files according to the instructions above.")
        return manager, [], {}, {}, {}


if __name__ == "__main__":
    manager, people, surfaces, density, viz = create_trial_people_demo()
    print("\n✅ Trial People Manager Demo completed!") 