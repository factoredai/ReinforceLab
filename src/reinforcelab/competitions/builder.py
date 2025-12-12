import os
import shutil
import yaml
import pathlib
import inspect
from dataclasses import dataclass, field
from typing import List, Optional

# ==========================
# Configuration Classes
# ==========================

@dataclass
class PhaseConfig:
    name: str
    env_id: str
    phase_type: str  # 'evaluation' or 'convergence'
    num_episodes: int = 100
    goal_reward: float = 0.0
    stability_window: int = 100
    max_steps: int = 100000
    num_runs: int = 5
    color: str = "green"
    start: str = "1-1-2024"  # Start date in M-D-YYYY format
    end: Optional[str] = None  # End date in M-D-YYYY format (optional)

@dataclass
class RLCompetition:
    title: str
    description: str
    docker_image: str = "codalab/codalab-legacy:py37"
    phases: List[PhaseConfig] = field(default_factory=list)
    
    def add_phase(self, phase: PhaseConfig):
        self.phases.append(phase)
        
    def build(self, build_dir="./build", override_dir: Optional[str] = None):
        # If no override_dir specified, use the caller's directory
        if override_dir is None:
            caller_frame = inspect.stack()[1]
            caller_file = caller_frame.filename
            override_dir = str(pathlib.Path(caller_file).parent)
        
        builder = CompetitionBuilder(self, build_dir, override_dir)
        builder.run()

# ==========================
# The Builder
# ==========================

class CompetitionBuilder:
    def __init__(self, config: RLCompetition, build_dir: str, override_dir: str):
        self.config = config
        self.build_dir = build_dir
        self.bundle_dir = os.path.join(build_dir, "bundle")
        self.starting_kit_dir = os.path.join(self.bundle_dir, "starting_kit")
        self.template_dir = pathlib.Path(__file__).parent / "templates"
        self.override_dir = pathlib.Path(override_dir)

    def _get_template_path(self, filename: str) -> Optional[pathlib.Path]:
        """Get the path to a template file, checking override_dir first."""
        # First check if an override exists in the competition folder
        override_path = self.override_dir / filename
        if override_path.exists():
            return override_path
        # Fall back to the default templates
        template_path = self.template_dir / filename
        if template_path.exists():
            return template_path
        return None

    def _read_template(self, filename: str) -> str:
        """Read a template file, with override_dir taking precedence."""
        path = self._get_template_path(filename)
        if path is None:
            return ""
        return path.read_text()

    def _write_file(self, dest_path: str, content: str):
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        with open(dest_path, "w") as f:
            f.write(content)

    def _render(self, filename: str, replacements: dict) -> str:
        content = self._read_template(filename)
        for key, value in replacements.items():
            # Try both quoted and unquoted placeholders (for Python vs Markdown)
            placeholder_quoted = f'"___{key}___"'
            placeholder_unquoted = f"___{key}___"
            if isinstance(value, str):
                content = content.replace(placeholder_quoted, f'"{value}"')
                content = content.replace(placeholder_unquoted, value)
            else:
                content = content.replace(placeholder_quoted, str(value))
                content = content.replace(placeholder_unquoted, str(value))
        return content

    def run(self):
        self._build_bundle()
        print(f"Build Complete. Upload '{os.path.abspath(self.bundle_dir)}.zip' to CodaBench.")

    def _build_bundle(self):
        # 1. Clean previous build
        if os.path.exists(self.bundle_dir): shutil.rmtree(self.bundle_dir)
        
        # 2. Setup Data Directories
        # CodaBench requires these folders to exist, even if empty.
        os.makedirs(os.path.join(self.bundle_dir, "input_data"), exist_ok=True)
        self._write_file(os.path.join(self.bundle_dir, "input_data", "placeholder.txt"), "no data needed")
        
        os.makedirs(os.path.join(self.bundle_dir, "reference_data"), exist_ok=True)
        self._write_file(os.path.join(self.bundle_dir, "reference_data", "gold.txt"), "no references needed")

        # 3. Build Scoring Program (Shared)
        scoring_dir = os.path.join(self.bundle_dir, "scoring_program")
        self._write_file(os.path.join(scoring_dir, "program.py"), self._read_template("scoring.py"))
        self._write_file(os.path.join(scoring_dir, "metadata.yaml"), "command: python program.py")

        # 4. Build Tasks (One Ingestion Program per Phase = One Task)
        tasks_list = []
        phases_list = []

        for i, phase in enumerate(self.config.phases):
            task_index = i
            p_idx = i + 1
            
            # Directory name for this task's ingestion
            ingestion_folder_name = f"ingestion_program_{p_idx}"
            ingestion_dir = os.path.join(self.bundle_dir, ingestion_folder_name)
            
            # --- Write Ingestion Code ---
            self._write_file(os.path.join(ingestion_dir, "monitor.py"), self._read_template("monitor.py"))
            
            if phase.phase_type == 'evaluation':
                code = self._render("ingestion_eval.py", {
                    "ENV_ID": phase.env_id,
                    "NUM_EPISODES": phase.num_episodes
                })
            elif phase.phase_type == 'convergence':
                code = self._render("ingestion_train.py", {
                    "ENV_ID": phase.env_id,
                    "GOAL": phase.goal_reward,
                    "WINDOW": phase.stability_window,
                    "MAX_STEPS": phase.max_steps,
                    "NUM_RUNS": phase.num_runs
                })

            self._write_file(os.path.join(ingestion_dir, "program.py"), code)
            self._write_file(os.path.join(ingestion_dir, "metadata.yaml"), "command: python program.py")

            # --- Define Task ---
            task_def = {
                "index": task_index,
                "name": f"{phase.name} Task",
                "description": f"Task for {phase.name}",
                "input_data": "input_data",         # points to folder at root
                "reference_data": "reference_data", # points to folder at root
                "ingestion_program": ingestion_folder_name, # points to folder at root
                "scoring_program": "scoring_program"        # points to folder at root
            }
            tasks_list.append(task_def)

            # --- Define Phase ---
            phase_def = {
                "index": task_index,
                "name": phase.name,
                "description": f"{phase.name} phase: {phase.phase_type} on {phase.env_id}",
                "start": phase.start,
                "max_submissions_per_day": 5,
                "max_submissions": 100,
                "execution_time_limit": 600,
                "tasks": [task_index],
                "solutions": []
            }
            if phase.end:
                phase_def["end"] = phase.end
            phases_list.append(phase_def)

        # 5. Extract phase information for page templates
        eval_phase = next((p for p in self.config.phases if p.phase_type == 'evaluation'), None)
        conv_phase = next((p for p in self.config.phases if p.phase_type == 'convergence'), None)
        
        # Default values if phases don't exist
        env_id = eval_phase.env_id if eval_phase else (conv_phase.env_id if conv_phase else "CartPole-v1")
        num_episodes = eval_phase.num_episodes if eval_phase else 100
        goal_reward = conv_phase.goal_reward if conv_phase else 0.0
        stability_window = conv_phase.stability_window if conv_phase else 100
        max_steps = conv_phase.max_steps if conv_phase else 100000
        num_runs = conv_phase.num_runs if conv_phase else 5

        # Generate title based on environment
        env_name = env_id.split('-')[0] if '-' in env_id else env_id
        competition_title = f"ReinforceLab: {env_name} Competition"
        
        # Generate short description
        description_parts = [
            f"Train and evaluate reinforcement learning agents on the {env_id} environment.",
            "Phase 1 evaluates pre-trained agents on average return over multiple episodes.",
            "Phase 2 measures training efficiency by tracking convergence time to target performance."
        ]
        competition_description = " ".join(description_parts)

        # 6. Generate competition.yaml
        # Using the exact structure from mini-automl
        yaml_content = {
            "version": 2,
            "title": competition_title,
            "docker_image": self.config.docker_image,
            "image": "logo.png",
            "description": competition_description,
            "terms": "pages/terms.md",       # In pages directory
            "registration_auto_approve": False,
            "pages": [
                {"title": "Overview", "file": "pages/overview.md"},
                {"title": "Evaluation", "file": "pages/evaluation.md"}
            ],
            "leaderboards": [
                {
                    "index": 0,
                    "title": "Results",
                    "key": "Results",
                    "submission_rule": "Force_Last",
                    "columns": [
                        {
                            "title": "Score",
                            "key": "score",
                            "index": 0,
                            "sorting": "desc"
                        }
                    ]
                }
            ],
            "starting_kit": "starting_kit",
            "tasks": tasks_list,
            "phases": phases_list,
            "solutions": [
                {
                    "index": 0,
                    "tasks": list(range(len(tasks_list))),  # Apply to all tasks
                    "path": "solution/"
                }
            ]
        }
        
        with open(os.path.join(self.bundle_dir, "competition.yaml"), "w") as f:
            yaml.dump(yaml_content, f, sort_keys=False)

        # 7. Copy Logo (check override_dir first)
        logo_path = self._get_template_path("logo.png")
        if logo_path:
            shutil.copy(logo_path, os.path.join(self.bundle_dir, "logo.png"))

        # 8. Create Sample Solution (Random Agent)
        solution_dir = os.path.join(self.bundle_dir, "solution")
        os.makedirs(solution_dir, exist_ok=True)
        self._write_file(
            os.path.join(solution_dir, "agent.py"),
            self._read_template("solution_random_agent.py")
        )

        # 9. Create Pages Directory and Render Templates
        pages_dir = os.path.join(self.bundle_dir, "pages")
        os.makedirs(pages_dir, exist_ok=True)
        
        # Render overview.md
        overview_content = self._render("overview.md", {
            "TITLE": competition_title,
            "DESCRIPTION": competition_description,
            "NUM_EPISODES": str(num_episodes),
            "ENV_ID": env_id,
            "GOAL_REWARD": str(goal_reward),
            "STABILITY_WINDOW": str(stability_window),
            "MAX_STEPS": str(max_steps),
            "NUM_RUNS": str(num_runs)
        })
        self._write_file(os.path.join(pages_dir, "overview.md"), overview_content)
        
        # Render evaluation.md
        evaluation_content = self._render("evaluation.md", {
            "NUM_EPISODES": str(num_episodes),
            "ENV_ID": env_id,
            "GOAL_REWARD": str(goal_reward),
            "STABILITY_WINDOW": str(stability_window),
            "MAX_STEPS": str(max_steps),
            "NUM_RUNS": str(num_runs)
        })
        self._write_file(os.path.join(pages_dir, "evaluation.md"), evaluation_content)
        
        # Render terms.md
        terms_content = self._read_template("terms.md")
        self._write_file(os.path.join(pages_dir, "terms.md"), terms_content)
        
        # 10. Build Starting Kit (inside bundle)
        self._build_starting_kit()
        
        # 11. Zip Bundle
        shutil.make_archive(os.path.join(self.build_dir, "bundle"), 'zip', self.bundle_dir)

    def _build_starting_kit(self):
        """Build the starting kit folder inside the bundle."""
        os.makedirs(self.starting_kit_dir, exist_ok=True)
        
        # Extract phase configuration
        eval_phase = next((p for p in self.config.phases if p.phase_type == 'evaluation'), None)
        conv_phase = next((p for p in self.config.phases if p.phase_type == 'convergence'), None)
        
        default_env = eval_phase.env_id if eval_phase else (conv_phase.env_id if conv_phase else "CartPole-v1")
        env_name = default_env.split('-')[0] if '-' in default_env else default_env
        competition_title = f"ReinforceLab: {env_name} Competition"
        
        # Phase configuration with defaults
        num_episodes = eval_phase.num_episodes if eval_phase else 100
        goal_reward = conv_phase.goal_reward if conv_phase else 0.0
        stability_window = conv_phase.stability_window if conv_phase else 100
        max_steps = conv_phase.max_steps if conv_phase else 100000
        num_runs = conv_phase.num_runs if conv_phase else 5
        
        # Copy logo to starting kit
        logo_path = self._get_template_path("logo.png")
        if logo_path:
            shutil.copy(logo_path, os.path.join(self.starting_kit_dir, "logo.png"))
        
        # Agent template and random agent implementation
        self._write_file(
            os.path.join(self.starting_kit_dir, "agent.py"),
            self._read_template("agent.py")
        )
        self._write_file(
            os.path.join(self.starting_kit_dir, "random_agent.py"),
            self._read_template("solution_random_agent.py")
        )
        self._write_file(
            os.path.join(self.starting_kit_dir, "monitor.py"),
            self._read_template("monitor.py")
        )
        
        # Run local script with full phase configuration
        run_local_code = self._render("run_local.py", {
            "ENV_ID": default_env,
            "NUM_EPISODES": str(num_episodes),
            "GOAL_REWARD": str(goal_reward),
            "STABILITY_WINDOW": str(stability_window),
            "MAX_STEPS": str(max_steps),
            "NUM_RUNS": str(num_runs)
        })
        self._write_file(os.path.join(self.starting_kit_dir, "run_local.py"), run_local_code)
        
        # Training script with environment and goal configured
        train_code = self._render("train.py", {
            "ENV_ID": default_env,
            "GOAL_REWARD": str(goal_reward)
        })
        self._write_file(os.path.join(self.starting_kit_dir, "train.py"), train_code)

        # Requirements (read from template or competition override)
        requirements_content = self._read_template("requirements.txt")
        self._write_file(os.path.join(self.starting_kit_dir, "requirements.txt"), requirements_content)
        
        # Sample submission folder with example agent and checkpoint
        sample_dir = os.path.join(self.starting_kit_dir, "sample_submission")
        os.makedirs(sample_dir, exist_ok=True)
        self._write_file(
            os.path.join(sample_dir, "agent.py"),
            self._read_template("sample_submission_agent.py")
        )
        self._write_file(
            os.path.join(sample_dir, "checkpoint.txt"),
            self._read_template("sample_checkpoint.txt")
        )
        self._write_file(
            os.path.join(sample_dir, "requirements.txt"),
            requirements_content
        )
        # Create zip of sample submission for easy upload
        shutil.make_archive(
            os.path.join(self.starting_kit_dir, "sample_submission"),
            'zip',
            sample_dir
        )
        
        # Getting started notebook with environment and title configured
        notebook_content = self._render("getting_started.ipynb", {
            "ENV_ID": default_env,
            "TITLE": competition_title
        })
        self._write_file(os.path.join(self.starting_kit_dir, "getting_started.ipynb"), notebook_content)
