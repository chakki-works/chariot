import pandas as pd
from joblib import Parallel, delayed
from chariot.transformer.base_preprocessor import BasePreprocessor
from chariot.preprocessor import Preprocessor
from chariot.base_processor import BaseProcessor


def _apply(input, output, process, dataframe):
    input_data = dataframe[input]
    if isinstance(process, (BasePreprocessor, Preprocessor)):
        transformed = process.transform(input_data)
    else:
        transformed = process.transform(input_data.values.reshape(-1, 1))
    return (output, transformed)


class Preprocess(BaseProcessor):

    def __init__(self, spec):
        super().__init__(spec)

    @classmethod
    def _make_tasks(cls, spec):
        input_output_process = []
        for input in spec:
            if isinstance(spec[input], dict):
                for output in spec[input]:
                    process = spec[input][output]
                    input_output_process.append((input, output, process))
            else:
                process = spec[input]
                input_output_process.append((input, input, process))
        return input_output_process

    def transform(self, dataframe, n_jobs=-1, as_dataframe=False):
        tasks = self._make_tasks(self.spec)
        _targets = list(self.spec.keys())

        output_transformed = Parallel(n_jobs=n_jobs)(
            delayed(_apply)(t[0], t[1], t[2], dataframe) for t in tasks)

        applied = {}
        for o_t in output_transformed:
            output, transformed = o_t
            applied[output] = transformed

        for c in dataframe.columns:
            if c not in _targets:
                applied[c] = dataframe[c]

        if not as_dataframe:
            return applied
        else:
            applied = pd.DataFrame.from_dict(applied)
            return applied
