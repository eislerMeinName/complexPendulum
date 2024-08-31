import numpy as np
import pandas as pd
from complexPendulum.assets import EvalSetup, EvaluationDataType, bcolors, RewardType


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

    def __init__(self, logpath: str, setups: list[EvalSetup],
                 datatype: EvaluationDataType, angle: float = 0.25,
                 epsilon: tuple = (0.02, 0.02)) -> None:
        """Initialization.

        Input:
            logpath: str
                The path to the file containing the log.
            setups: list[EvalSetup]
                The list of setups an agent should be evaluated on.
            datatype: EvaluationDataType
                The type of the evaluated data in order to cut the logged data into relevant part.
            epsilon: tuple
                The parameters for determining the settling time (x, theta).
        """

        print(bcolors.OKBLUE + f"Evaluating {datatype.name} \n" + bcolors.ENDC)
        self.logpath = logpath
        self.datatype = datatype
        self.Ts = 0
        self.index = None
        self.states, self.pwm, self.force, self.Time = loadLog(logpath)
        self.cutData(angle)
        self.setups = setups
        self.epsilon = epsilon
        self.data = {}
        _ = self.evalReturn()
        if datatype is EvaluationDataType.STEP:
            _ = self.evalStep()
        if datatype is EvaluationDataType.SWING_UP:
            _ = self.evalMaxima()

    def cutData(self, angle: float) -> None:
        """Cuts the data based on the EvaluationDataType and angle.
        Input:
            angle: float
                The absolute angle where swing-up is achieved.
        """

        for i in range(0, len(self.states)):
            if -angle <= self.states[i][0, 2] <= angle:
                self.index = i if i != 0 else None
                break

        match self.datatype:
            case EvaluationDataType.SWING_UP:
                if self.index is None:
                    return
                else:
                    self.Ts = self.Time[self.index - 1]
                    self.states = self.states[0:self.index]
                    self.pwm = self.pwm[0:self.index]
                    self.force = self.force[0:self.index]
                    self.Time = self.Time[0:self.index]

            case EvaluationDataType.STEP:
                if self.index is None:
                    return
                else:
                    self.Ts = self.Time[self.index - 1]
                    self.states = self.states[self.index:len(self.states)]
                    self.pwm = self.pwm[self.index:len(self.pwm)]
                    self.force = self.force[self.index:len(self.force)]
                    self.Time = self.Time[self.index:len(self.Time)]
            case EvaluationDataType.COMPLETE:
                if self.index is not None:
                    self.Ts = self.Time[self.index - 1]

    def evalMaxima(self) -> dict:
        """Evaluates the state maxima of swing up."""
        X = [abs(s[0, 0]) for s in self.states]
        Xdot = [abs(s[0, 1]) for s in self.states]
        Theta = [abs(s[0, 2]) for s in self.states]
        Thetadot = [abs(s[0, 3]) for s in self.states]
        self.data['|X'] = max(X)
        self.data['|X´|'] = max(Xdot)
        self.data['|θ|'] = max(Theta)
        self.data['|θ´|'] = max(Thetadot)

        return self.data

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

    def evalLIN(self, Q: np.array, R: np.array) -> float:
        """Evaluates the logged episode with undiscounted linear return.
        Return:
            lin: float
                The undiscounted linear return.
        """
        lin: float = 0
        for i, s in enumerate(self.states):
            lin -= np.linalg.norm(Q @ s.T) + np.linalg.norm(R * self.force[i])

        return lin

    def evalStep(self) -> dict:
        """Evaluates the logged data on different performance criteria."""

        xstart = abs(self.states[0][0, 0])
        thetastart = abs(self.states[0][0, 2])
        xreached: bool = False
        thetareached: bool = False
        indexTrTheta = None
        indexTrX = None
        hx = None
        htheta = None
        indexTmTheta = None
        indexTmX = None
        indexTeTheta = None
        indexTeX = None
        for i, state in enumerate(self.states):

            # check rise time
            if abs(state[0, 0]) < 0.1 * xstart and indexTrX is None:
                indexTrX = i
            if abs(state[0, 2]) < 0.1 * thetastart and indexTrTheta is None:
                indexTrTheta = i

            # check reached
            if not xreached and ((self.states[i - 1][0, 0] < 0 < state[0, 0]) or (
                    state[0, 0] < 0 < self.states[i - 1][0, 0])) and not i == 0:
                xreached = True
            if not thetareached and ((self.states[i - 1][0, 2] < 0 < state[0, 2]) or (
                    state[0, 2] < 0 < self.states[i - 1][0, 2])) and not i == 0:
                thetareached = True

            # peak time and overshoot
            if xreached and hx is None:
                X = [s[0, 0] for s in self.states]
                hx = max(X[i:len(X)])
                indexTmX = X.index(hx)
            if thetareached and htheta is None:
                Theta = [s[0, 2] for s in self.states]
                htheta = max(Theta[i:len(Theta)])
                indexTmTheta = Theta.index(htheta)

            # check settling time
            if abs(state[0, 0]) < self.epsilon[0] and indexTeX is None:
                X = [s[0, 0] for s in self.states]
                if max(X[i:len(X)]) < self.epsilon[0] and min(X[i:len(X)]) > -self.epsilon[0]:
                    indexTeX = i
            if abs(state[0, 2]) < self.epsilon[1] and indexTeTheta is None:
                Theta = [s[0, 2] for s in self.states]
                if max(Theta[i:len(Theta)]) < self.epsilon[1] and min(Theta[i: len(Theta)]) > -self.epsilon[1]:
                    indexTeTheta = i

        eXinf = self.states[-1][0, 0]
        eThetainf = self.states[-1][0, 2]
        self.data["X: Tr"] = self.Time[indexTrX]
        self.data["X: Tm"] = self.Time[indexTmX]
        self.data['X: Δh'] = hx
        self.data["X: Te"] = self.Time[indexTeX] if indexTeX is not None else None
        self.data['X: e∞'] = eXinf

        self.data['θ: Tr'] = self.Time[indexTrTheta]
        self.data['θ: Tm'] = self.Time[indexTmTheta]
        self.data['θ: Δh'] = htheta
        self.data['θ: Te'] = self.Time[indexTeTheta] if indexTeTheta is not None else None
        self.data['θ: e∞'] = eThetainf

        return self.data

    def evalReturn(self) -> dict:
        """Evaluates the logged data on all setups."""
        self.data["name"] = self.logpath.replace('../', '').replace('.csv', '')
        for setup in self.setups:
            match setup.func:
                case RewardType.LQ:
                    self.data[setup.name] = self.evalLQR(setup.Q, setup.R, setup.k)
                case RewardType.EXP:
                    self.data[setup.name] = self.evalEXP(setup.Q, setup.R)
                case RewardType.LIN:
                    self.data[setup.name] = self.evalLIN(setup.Q, setup.R)
        self.data["Ts"] = self.Ts
        return self.data

    def __str__(self):
        """Prints evaluation as Dataframe."""
        return "\n".join("{}:\t{}".format(k, v) for k, v in self.data.items())
