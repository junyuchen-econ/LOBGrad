import yaml
from datetime import datetime


# ======================================
# Class: Config
# ======================================
class Config:
    def __init__(self, file: str = 'config.yaml'):
        with open(file, 'r') as f:
            data = yaml.safe_load(f)

        # Data
        self.symbol: str = data['symbol']
        self.price_level: int = data['price_level']
        self.level: int = data['level']
        self.time_diff: int = data['time_diff']
        self.depth: int = data['depth']

        # Model
        self.beta_start: float = data['beta_start']
        self.beta_end: float = data['beta_end']
        self.schedule: str = data['schedule']
        self.context_length: int = data['context_length']
        self.prediction_length: int = data['prediction_length']
        self.input_dim: int = 2 * data['level'] + 1  # +1 for midprice
        self.hidden_encoder_dim: int = data['hidden_encoder_dim']
        self.hidden_noise_dim: int = data['hidden_noise_dim']
        self.num_encoder_layers: int = data['num_encoder_layers']
        self.emb_dim: int = data['emb_dim']
        self.k: int = data['k']
        self.conditioning_length: int = data.get('conditioning_length', None)
        self.loss_type: str = data.get('loss_type', 'l2')
        self.dropout_rate: float = data.get('dropout_rate', 0.0)
        self.scaling: bool = data.get('scaling', False)
        self.cell_type: str = data.get('cell_type', 'GRU')
        self.noise_predictor_type: str = data.get('noise_predictor_type', 'mlp')
        self.residual_layers: int = data.get('residual_layers', 8)
        self.residual_channels: int = data.get('residual_channels', 8)
        self.dilation_cycle_length: int = data.get('dilation_cycle_length', 2)

        # Reproducibility
        self.seed: int = data.get('seed', 42)

        # Training
        self.batch_size: int = data['batch_size']
        self.learning_rate: float = data['learning_rate']
        self.num_epochs: int = data['num_epochs']
        self.max_batches: int = data['max_batches']
        self.max_grad_norm: float | None = data.get('max_grad_norm', None)

        # Prediction
        self.prediction_steps: int = data['prediction_steps']
        self.num_generated_paths: int = data['num_generated_paths']
        self.start_index: int = data['start_index']

        # Dates
        self.start_date_train = datetime.strptime(data['train']['start'], '%Y-%m-%d')
        self.end_date_train = datetime.strptime(data['train']['end'], '%Y-%m-%d')
        self.start_date_val = datetime.strptime(data['val']['start'], '%Y-%m-%d')
        self.end_date_val = datetime.strptime(data['val']['end'], '%Y-%m-%d')
        self.start_date_test = datetime.strptime(data['test']['start'], '%Y-%m-%d')
        self.end_date_test = datetime.strptime(data['test']['end'], '%Y-%m-%d')


config = Config()
