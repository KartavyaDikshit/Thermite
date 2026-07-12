import os
from datetime import datetime

class ModelCard:
    """Utility to automatically generate Model Cards for trained models."""
    
    @staticmethod
    def generate(model, filepath="model_card.md"):
        """Generate a markdown model card.
        
        Parameters
        ----------
        model : object
            The trained model instance.
        filepath : str
            The path where the model card will be saved.
        """
        model_name = model.__class__.__name__
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Extract basic parameters (public attributes not starting with _)
        params = {}
        for k, v in vars(model).items():
            if not k.startswith("_"):
                params[k] = v
                
        # Generate markdown content
        lines = [
            f"# Model Card: {model_name}",
            "",
            f"**Generated on:** {timestamp}",
            "",
            "## Model Architecture",
            f"- **Type:** {model_name}",
            "",
            "## Hyperparameters"
        ]
        
        if params:
            for k, v in params.items():
                lines.append(f"- **{k}**: {v}")
        else:
            lines.append("- *No public hyperparameters found.*")
            
        lines.append("")
        lines.append("## Intended Use")
        lines.append("This model is automatically documented by Thermite Auto-Docs.")
        
        with open(filepath, "w") as f:
            f.write("\n".join(lines))
