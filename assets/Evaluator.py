import numpy as np
import pandas as pd
from assets import Setup1, Setup2, Setup3, Setup4, Setup5, Setup6, Setup7, EvalSetup, SetupType, EvaluationDataType


def loadLog(path: str) -> tuple:
    """Loading method that loads logged Data.
    Input:
        path: str
            The path to the log file in csv format.

    Return:
        states: list[np.array]
            The logged states relevant for evaluation.
        pwm: list[float]
            The logged pwm signals.
        force: list[float]
            The logged applied forces.
    """

    frame = pd.read_csv(path)
    X = frame['X'].to_list()
    Xdot = frame['Xdot'].to_list()
    Theta = frame['Theta'].to_list()
    Thetadot = frame['Thetadot'].to_list()
    pwm = frame['pwm'].to_list()
    force = frame['force'].to_list()
    X.pop(0)
    Xdot.pop(0)
    Theta.pop(0)
    Thetadot.pop(0)
    pwm.pop(0)
    force.pop(0)
    states = [np.array([[X[i], Xdot[i], Theta[i], Thetadot[i]]]) for i in range(0, len(X))]
    return states, pwm, force, frame['Time'].to_list()


class Evaluator:
    """Evaluator class that allows evaluation of a logged, successful episode with respect to different reward functions."""

    def __init__(self, logpath: str, evalpath: str, setups: list[EvalSetup],
                 datatype: EvaluationDataType, angle: float = 0.25) -> None:
        """Initialization.

        Input:
            logpath: str
                The path to the file containing the log.
            evalpath: str
                The path to the file for writing the evaluation.
            setups: list[EvalSetup]
                The list of setups an agent should be evaluated on.
            datatype: EvaluationDataType
                The type of the evaluated data in order to cut the logged data into relevant part.
        """

        self.logpath = logpath
        self.datatype = datatype
        self.index = None
        self.states, self.pwm, self.force, self.Time = loadLog(logpath)
        self.cutData(angle)
        self.evalpath = evalpath
        self.setups = setups
        self.data = {}
        _ = self.evalReturn()
        if datatype is EvaluationDataType.STEP:
            _ = self.evalStep()

    def cutData(self, angle: float) -> None:
        """Cuts the data based on the EvaluationDataType and angle.
        Input:
            angle: float
                The absolute angle where swing-up is achieved.
        """

        for i in range(0, len(self.states)):
            if -angle <= self.states[i][0, 2] <= angle:
                self.index = i
                break
        print("Swing-Up took ~{0}s".format(self.Time[self.index + 1]))
        match self.datatype:
            case EvaluationDataType.SWING_UP:
                if self.index is None:
                    return
                else:
                    self.states = self.states[0:self.index]
                    self.pwm = self.pwm[0:self.index]
                    self.force = self.force[0:self.index]

            case EvaluationDataType.STEP:
                if self.index is None:
                    return
                else:
                    self.states = self.states[self.index:len(self.states)]
                    self.pwm = self.pwm[self.index:len(self.pwm)]
                    self.force = self.force[self.index:len(self.force)]

    def evalLQR(self, Q: np.array, R: np.array, k: float = 1) -> float:
        """Evaluates the log with undiscounted linear quadratic return.
        Input:
            Q: np.array
                The Q matrix.
            R: np.array
                The R matrix.
            k: float
                The constant normalization factor.

        Return:
            lqr: float
                The undiscounted linear quadratic return of the logged episode.
        """

        lqr: float = 0
        for i, s in enumerate(self.states):
            lqr -= (s @ Q @ s.T + self.force[i] * R * self.force[i])[0, 0] / k

        return lqr

    def evalEXP(self, Q: np.array, R: np.array) -> float:
        """Evaluates the logged episode with undiscounted exponential return.
        Input:
            Q: np.array
                The Q matrix.
            R: np.array
                The R matrix.

        Return:
            expo: float
                The undiscounted exponential return.
        """

        expo: float = 0
        for i, s in enumerate(self.states):
            expo += np.exp(-np.linalg.norm(Q @ s.T)) + np.exp(- np.linalg.norm(R * self.force[i])) - 2

        return expo

    def evalLIN(self) -> float:
        """Evaluates the logged episode with undiscounted linear return.
        Return:
            lin: float
                The undiscounted linear return.
        """
        lin: float = 0
        for s in self.states:
            lin -= abs(s[0, 0]) + abs(s[0, 2])

        return lin

    def evalStep(self) -> None:
        """Evaluates the logged data on different performance criteria."""

        # TODO
        pass

    def evalReturn(self) -> dict:
        """Evaluates the logged data on all setups."""
        self.data["name"] = self.logpath.replace('../', '').replace('.csv', '')
        for setup in self.setups:
            match setup.func:
                case SetupType.LQR:
                    self.data[setup.name] = self.evalLQR(setup.Q, setup.R, setup.k)
                case SetupType.EXP:
                    self.data[setup.name] = self.evalEXP(setup.Q, setup.R)
                case SetupType.LIN:
                    self.data[setup.name] = self.evalLIN()
        self.data["Ts"] = self.Time[self.index + 1]
        return self.data

    def __str__(self):
        """Prints evaluation as Dataframe."""
        return "\n".join("{}:\t{}".format(k, v) for k, v in self.data.items())


if __name__ == "__main__":
    setups = [Setup1, Setup2, Setup3, Setup4, Setup5, Setup6, Setup7]
    eval = Evaluator("../test.csv", "", setups, EvaluationDataType.COMPLETE)
    print(eval)
