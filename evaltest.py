import numpy as np
from complexPendulum.assets import Evaluator, Setup1, Setup2, Setup3, Setup4, Setup5, Setup6, Setup7, EvaluationDataType

if __name__ == "__main__":
    setups = [Setup1, Setup2, Setup3, Setup4, Setup5, Setup6, Setup7]
    evaluator = Evaluator("test.csv", setups, EvaluationDataType.STEP, epsilon=(0.03, np.pi/1024))
    print(evaluator)
