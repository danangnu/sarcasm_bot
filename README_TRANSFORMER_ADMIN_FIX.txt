TRANSFORMER ADMIN FIX

Replace:
  app.py

Add:
  training/run_transformer_retraining.py

Required existing file:
  training/train_transformer_model.py

Candidate output:
  models/transformer_candidate

Promoted active model:
  models/transformer_model

The existing BiLSTM files are not deleted and remain available as rollback.
