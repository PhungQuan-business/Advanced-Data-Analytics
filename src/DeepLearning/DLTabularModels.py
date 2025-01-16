import torch
from torch import nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from DeepLearning.lossFunction import (
    DiceLoss,
    DiceBCELoss,
    IoULoss,
    FocalLoss,
    TverskyLoss,
    FocalTverskyLoss,
)

class TabularModels():
    def __init__(self, target_col, num_col_names, cat_col_names, epoch, algorithm):
        self.target_col = target_col
        self.num_col_names = num_col_names
        self.cat_col_names = cat_col_names
        self.epoch = epoch
        self.algorithm = algorithm
    
    def NODE(self):
        from pytorch_tabular.models import NodeConfig
        from pytorch_tabular.config import (
            DataConfig,
            OptimizerConfig,
            TrainerConfig,
        )
        from pytorch_tabular import TabularModel

        node_data_config = DataConfig(
            target=[
                self.target_col
            ],  # target should always be a list
            continuous_cols=self.num_col_names.columns.tolist(),
            # categorical_cols=cat_col_names,
        )

        node_config = NodeConfig(
            task="classification",               # Task: 'classification' or 'regression'
            num_layers=3,                        # Number of layers in the architecture ### Sửa cái này được
            num_trees=512,                       # Number of trees in each layer ### Sửa cái này được
            additional_tree_output_dim=2,        # Additional output dimensions passed across layers ### Sửa cái này được
            depth=6,                             # Depth of each tree
            choice_function="entmax15",          # Choice function for soft feature selection: 'entmax15' or 'sparsemax'
            bin_function="entmoid15",            # Bin function for sparse leaf weights: 'entmoid15' or 'sparsemoid'
            max_features=None,                   # Limit on features carried forward; default = None
            input_dropout=0.1,                   # Dropout applied to inputs between layers
            initialize_response="normal",        # Initialization method for response variable
            initialize_selection_logits="uniform",  # Initialization for feature selectors
            threshold_init_beta=1.0,             # Beta parameter for data-aware threshold initialization
            threshold_init_cutoff=1.0,           # Cutoff for threshold initialization
            embedding_dims=None,                 # Infer embedding dimensions automatically
            embedding_dropout=0.1,               # Dropout for categorical embeddings
            batch_norm_continuous_input=True,    # Batch normalization for continuous features
            learning_rate=1e-3,                  # Learning rate
            metrics=["accuracy", "precision", "recall", "f1_score", "auroc"],
            metrics_prob_input=[True, True, True, True, True],           # Input probabilities for accuracy metric
            seed=42                              # Seed for reproducibility
        )

        node_optimizer_config = OptimizerConfig()

        node_trainer_config = TrainerConfig(
            auto_lr_find=False,
            batch_size=512, # có thể để batch nhỏ vì dữ liệu nhỏ
            max_epochs=self.epoch, # có thể để nhu này và giảm batch vì dữ liệu nhỏ
        )

        tabular_model = TabularModel(
            data_config=node_data_config,
            model_config=node_config,
            optimizer_config=node_optimizer_config,
            trainer_config=node_trainer_config,
            verbose=True
        )
        return tabular_model
        
    def TabNet(self):
        from pytorch_tabular.models import TabNetModelConfig
        from pytorch_tabular import TabularModel
        from pytorch_tabular.config import (
            DataConfig,
            OptimizerConfig,
            TrainerConfig,
        )
    
        tabnet_data_config = DataConfig(
            target=[
                self.target_col
            ],  # target should always be a list
            continuous_cols=self.num_col_names.columns.tolist(),
            categorical_cols=self.cat_col_names,
        )

        tabnet_config = TabNetModelConfig(
            n_d=16,                          # Dimension of prediction layer
            n_a=16,                          # Dimension of attention layer
            n_steps=5,                       # Number of successive steps in the network
            gamma=1.5,                       # Scaling factor for attention updates
            n_independent=2,                 # Number of independent GLU layers
            n_shared=2,                      # Number of shared GLU layers
            virtual_batch_size=128,          # Batch size for Ghost Batch Normalization
            mask_type="entmax",              # Masking function: 'sparsemax' or 'entmax'
            task="classification",           # Task type: classification or regression
            learning_rate=1e-3,              # Learning rate
            metrics=["accuracy", "precision", "recall", "f1_score", "auroc"],  # Metrics to track
            metrics_prob_input=[True, True, True, True, True],           # Probabilities as input
            seed=42                          # Seed for reproducibility
        )

        tabnet_optimizer_config = OptimizerConfig()

        tabnet_trainer_config = TrainerConfig(
            auto_lr_find=False,
            batch_size=512,
            max_epochs=self.epoch,
        )

        tabular_model = TabularModel(
            data_config=tabnet_data_config,
            model_config=tabnet_config,
            optimizer_config=tabnet_optimizer_config,
            trainer_config=tabnet_trainer_config,
            verbose=True
        )
        
        return tabular_model
    
    def AutoInt(self):
        from pytorch_tabular.models import AutoIntConfig
        from pytorch_tabular import TabularModel
        from pytorch_tabular.config import (
            DataConfig,
            OptimizerConfig,
            TrainerConfig,
        )

        autoint_data_config = DataConfig(
            target=[
                self.target_col
            ],  # target should always be a list
            continuous_cols=self.num_col_names.columns.tolist(),
            categorical_cols=self.cat_col_names,
        )


        autoint_config = AutoIntConfig(
            attn_embed_dim=32,            # Hidden units in Multi-Head Attention layers
            num_heads=2,                  # Number of attention heads
            num_attn_blocks=3,            # Number of attention layers
            attn_dropouts=0.1,            # Dropout between attention layers
            has_residuals=True,           # Enable residual connections
            embedding_dim=14,             # Embedding dimensions for features
            embedding_initialization="kaiming_normal",  # Embedding initialization scheme
            embedding_bias=True,          # Include bias in embedding layers
            share_embedding=False,        # Disable shared embeddings
            deep_layers=True,             # Enable deep MLP layers
            layers="14-14-14",           # Deep MLP layer configuration
            activation="ReLU",            # Activation function in deep MLP
            use_batch_norm=True,          # Include BatchNorm after layers
            dropout=0.1,                  # Dropout probability
            attention_pooling=False,      # Combine attention outputs for final prediction
            task="classification",        # Task type: regression or classification
            head="LinearHead",            # Prediction head type
            learning_rate=1e-3,           # Learning rate
            metrics=["accuracy", "f1_score", "precision", "recall", "auroc"],  # Metrics to track
            # metrics_params=[{}, {}, {}, {}, {"task": "multiclass"}],  # Metrics parameters
            metrics_prob_input=[True, True, True, True, True],  # Metrics expect probabilities
            seed=42,                      # Random seed for reproducibility
        )

        autoint_optimizer = OptimizerConfig()

        autoint_trainer_config = TrainerConfig(
            auto_lr_find=False,
            batch_size=512,
            max_epochs=self.epoch
        )

        tabular_model = TabularModel(
            data_config=autoint_data_config,
            model_config=autoint_config,
            optimizer_config=autoint_optimizer,
            trainer_config=autoint_trainer_config,
        )
        
        return tabular_model
    
    def TabTransformer(self):
        from pytorch_tabular.models import TabTransformerConfig
        from pytorch_tabular.config import (
            DataConfig,
            OptimizerConfig,
            TrainerConfig,
        )

        tab_transformer_data_config = DataConfig(
            target=[
                self.target_col
            ],  # target should always be a list
            continuous_cols=self.num_col_names.columns.tolist(),
            categorical_cols=self.cat_col_names,
        )

        tab_transformer_config = TabTransformerConfig(
            input_embed_dim=32,                # Embedding dimension for categorical features
            embedding_initialization="kaiming_normal",  # Embedding initialization scheme
            embedding_bias=False,              # Disable embedding bias
            share_embedding=False,             # Disable shared embeddings
            num_heads=8,                       # Number of attention heads
            num_attn_blocks=6,                 # Number of transformer blocks
            transformer_head_dim=None,         # Default to same as `input_embed_dim`
            attn_dropout=0.1,                  # Dropout after Multi-Headed Attention
            add_norm_dropout=0.1,              # Dropout in AddNorm layers
            ff_dropout=0.1,                    # Dropout in Feed-Forward layers
            ff_hidden_multiplier=4,           # Multiplier for feed-forward hidden layers
            transformer_activation="GEGLU",    # Activation function in transformer layers
            task="classification",             # Task type: regression or classification
            head="LinearHead",                 # Prediction head type
            learning_rate=1e-3,                # Learning rate
            # loss="cross_entropy",              # Loss function for classification
            metrics=["accuracy", "f1_score", "precision", "recall", "auroc"],  # Metrics to track
            # metrics_params=[{}, {}, {}, {}, {"task": "multiclass"}],  # Metrics parameters
            metrics_prob_input=[True, True, True, True, True],  # Metrics expect probabilities
            seed=42                            # Random seed for reproducibility
        )

        tab_transformer_optimizer = OptimizerConfig()

        tab_transformer_trainer_config = TrainerConfig(
            auto_lr_find=False,
            batch_size=512,
            max_epochs=self.epoch
        )
        from pytorch_tabular import TabularModel

        tabular_model = TabularModel(
            data_config=tab_transformer_data_config,
            model_config=tab_transformer_config,
            optimizer_config=tab_transformer_optimizer,
            trainer_config=tab_transformer_trainer_config
        )
        
        return tabular_model
            
    def GATE(self):
        from pytorch_tabular.models import GatedAdditiveTreeEnsembleConfig
        from pytorch_tabular import TabularModel
        from pytorch_tabular.config import (
            DataConfig,
            OptimizerConfig,
            TrainerConfig,
        )

        gate_data_config = DataConfig(
            target=[
                self.target_col
            ],  # target should always be a list
            continuous_cols=self.num_col_names.columns.tolist(),
            categorical_cols=self.cat_col_names,
        )

        gate_config = GatedAdditiveTreeEnsembleConfig(
            gflu_stages=6,                       # Number of feature abstraction layers
            gflu_dropout=0.1,                    # Dropout rate for abstraction layers
            tree_depth=5,                        # Depth of each tree
            num_trees=20,                        # Number of trees in the ensemble
            binning_activation="sparsemoid",     # Activation function for binning
            feature_mask_function="sparsemax",   # Feature mask function
            tree_dropout=0.1,                    # Dropout probability in tree binning
            chain_trees=True,                    # Chain trees (boosting) or bag trees (parallel)
            tree_wise_attention=True,            # Enable tree-wise attention
            tree_wise_attention_dropout=0.1,     # Dropout in tree-wise attention
            share_head_weights=True,             # Share weights across heads
            task="classification",               # Task type: regression or classification
            head="LinearHead",                   # Prediction head type
            learning_rate=1e-3,                  # Learning rate
            metrics=["accuracy", "f1_score", "precision", "recall", "auroc"],  # Metrics to track
            # metrics_params=[{}, {}, {}, {}, {"task": "multiclass"}],  # Metrics parameters
            metrics_prob_input=[True, True, True, True, True],  # Metrics expect probabilities
            seed=42                              # Seed for reproducibility
        )

        gate_optimizer_config = OptimizerConfig()

        gate_trainer_config = TrainerConfig(
            auto_lr_find=False,
            batch_size=512,
            max_epochs=self.epoch
        )

        tabular_model = TabularModel(
            data_config=gate_data_config,
            model_config=gate_config,
            optimizer_config=gate_optimizer_config,
            trainer_config=gate_trainer_config,
        )
        
        return tabular_model
    
    def GANDAF(self):
        from pytorch_tabular.models import GANDALFConfig
        from pytorch_tabular import TabularModel
        from pytorch_tabular.config import (
            DataConfig,
            OptimizerConfig,
            TrainerConfig,
        )

        gandaf_data_config = DataConfig(
            target=[
                self.target_col
            ],  # target should always be a list
            continuous_cols=self.num_col_names.columns.tolist(),
            categorical_cols=self.cat_col_names,
        )

        gandaf_config = GANDALFConfig(
            task="classification",
            gflu_stages=6,
            gflu_feature_init_sparsity=0.3,
            gflu_dropout=0.0,
            learning_rate=1e-3,
            metrics=["accuracy", "f1_score", "precision", "recall", "auroc"],  # Metrics to track
            # metrics_params=[{}, {}, {}, {}, {"task": "multiclass"}],  # Metrics parameters
            metrics_prob_input=[True, True, True, True, True],  # Metrics expect probabilities
        )

        gandaf_optimizer_config = OptimizerConfig()

        gandaf_trainer_config = TrainerConfig(
            batch_size=1024,
            max_epochs=self.epoch,
        )

        tabular_model = TabularModel(
            data_config=gandaf_data_config,
            model_config=gandaf_config,
            optimizer_config=gandaf_optimizer_config,
            trainer_config=gandaf_trainer_config,
            verbose=True
        )
        
        return tabular_model

    def DANETs(self):
        from pytorch_tabular.models import DANetConfig
        from pytorch_tabular import TabularModel
        from pytorch_tabular.config import (
            DataConfig,
            OptimizerConfig,
            TrainerConfig,
        )

        danet_data_config = DataConfig(
            target=[
                self.target_col
            ],  # target should always be a list
            continuous_cols=self.num_col_names.columns.tolist()
        )

        danet_config = DANetConfig(
            n_layers=8,                           # Number of Blocks in the DANet
            abstlay_dim_1=32,                     # Dimension of intermediate output in the first ABSTLAY layer
            abstlay_dim_2=64,                     # Dimension of intermediate output in the second ABSTLAY layer
            k=5,                                  # Number of feature groups in ABSTLAY layer
            dropout_rate=0.1,                     # Dropout in the Block
            task="classification",                # Task type: regression or classification
            head="LinearHead",                    # Prediction head type
            learning_rate=1e-3,                   # Learning rate
            metrics=["accuracy", "f1_score", "precision", "recall", "auroc"],  # Metrics to track
            # metrics_params=[{}, {}, {}, {}, {"task": "multiclass"}],  # Metrics parameters
            metrics_prob_input=[True, True, True, True, True],  # Metrics expect probabilities
            seed=42                               # Seed for reproducibility
        )

        danet_optimizer = OptimizerConfig()

        danet_trainer_config = TrainerConfig(
            auto_lr_find=False,
            batch_size=512,
            max_epochs=self.epoch
        )

        tabular_model = TabularModel(
            data_config=danet_data_config,
            model_config=danet_config,
            optimizer_config=danet_optimizer,
            trainer_config=danet_trainer_config,
        )
        
        return tabular_model
    
    def _get_loss_function(self, loss_function):
        if loss_function == 'DiceLoss':
            return DiceLoss()
        if loss_function == 'DiceBCELoss':
            return DiceBCELoss()
        if loss_function == 'IoULoss':
            return IoULoss()
        if loss_function == 'FocalLoss':
            return FocalLoss()
        if loss_function == 'TverskyLoss':
            return TverskyLoss()
        if loss_function == 'FocalTverskyLoss':
            return FocalTverskyLoss()
        if loss_function == 'ComboLoss':
            return ComboLoss()
        
    def _result_evaluation(self, train, val, test, algorithm, loss_function):
        loss = self._get_loss_function(loss_function)
        algorithm.fit(train=train, validation=val, loss=loss)
        result = algorithm.evaluate(test)
        
        return result
    
    def fit_predict(self, train, val, test, algorithm, loss_function):
        if self.algorithm == 'NODE':
            algorithm = self.NODE()
            result = self._result_evaluation(train, val, test, algorithm, loss_function)
            return result
        elif self.algorithm == 'TabNet':
            algorithm = self.TabNet()
            result = self._result_evaluation(train, val, test, algorithm, loss_function)
            return result
        elif self.algorithm == 'AutoInt':
            algorithm = self.AutoInt()
            result = self._result_evaluation(train, val, test, algorithm, loss_function)
            return result
        elif self.algorithm == 'TabTransformer':
            algorithm = self.TabTransformer()
            result = self._result_evaluation(train, val, test, algorithm, loss_function)
            return result
        elif self.algorithm == 'GATE':
            algorithm = self.GATE()
            result = self._result_evaluation(train, val, test, algorithm, loss_function)
            return result
        elif self.algorithm == 'GANDAF':
            algorithm = self.GANDAF()
            result = self._result_evaluation(train, val, test, algorithm, loss_function)
            return result
        elif self.algorithm == 'DANETs':
            algorithm = self.DANETs()
            result = self._result_evaluation(train, val, test, algorithm, loss_function)
            return result