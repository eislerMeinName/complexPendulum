from complexPendulum.assets import Evaluator, Setup1, Setup2, Setup3, Setup4, Setup5, Setup6, Setup7, EvaluationDataType

if __name__ == "__main__":
    setups = [Setup1, Setup2, Setup3, Setup4, Setup5, Setup6, Setup7]
    evaluator = Evaluator("test.csv", setups, EvaluationDataType.STEP)
    print(evaluator)
