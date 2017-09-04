# What Happens When Run t2t-trainer

Because they haven't written it yet.

1. Run command that looks something like:
```
t2t-trainer \
  --data_dir="${DATA_DIR}" \
  --problems="${PROBLEM}" \
  --model="${MODEL}" \
  --hparams_set="${HPARAMS}" \
  --output_dir="${TRAIN_DIR}"
```

2. The t2t-trainer file handles FLAGS extremely opaquely/sloppily, eventually calling:
```
trainer_utils.run(
        data_dir=data_dir,
        model=FLAGS.model,
        output_dir=output_dir,
        train_steps=FLAGS.train_steps,
        eval_steps=FLAGS.eval_steps,
        schedule=FLAGS.schedule)
``` 
which describes itself (`run`) as "Runs an Estimator locally or distributed." (located in `utils/trainer_utils.py`)

3. The `make_experiment` function in `trainer_utils.py` is called, which just returns a wrapper around `create_experiment`. Then `create_experiment` actually does some stuff:
    - Assigns `hparams` via the `create_hparams` function. 
    - Assigns `estimator` and `input_fns` via the `create_experiment_components` function.
    - Assigns `eval_metrics` via `metrics.create_evaluation_metrics`. 
Eventually, returns a `tf.contrib.learn.Experiment` instance, which soon thereafter calls its [train_and_evaluate](https://www.tensorflow.org/api_docs/python/tf/contrib/learn/Experiment#train_and_evaluate) method, and there she goes.
4. I have no idea what happens next.
