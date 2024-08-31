import numpy as np
from complexPendulum.assets import Evaluator, Setup1, Setup2, Setup3, Setup4, Setup5, Setup6, EvaluationDataType

EVAL_TYPE: EvaluationDataType = EvaluationDataType.COMPLETE


if __name__ == "__main__":
    setups = [Setup1, Setup2, Setup3, Setup4, Setup5, Setup6]
    evaluator = Evaluator("test.csv", setups, EVAL_TYPE, epsilon=(0.03, np.pi/1024))
    print(evaluator)
