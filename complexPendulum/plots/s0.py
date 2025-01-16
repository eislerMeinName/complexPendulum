import pandas as pd
import numpy as np

def calcS0(df) -> dict:
    """Mehhod that calculates the distribution based on a dataframe
    Input: 
        df: pandas dataframe
            The data.

    Returns:
        val: dict
            Dict with mean and std of each state parameter.

    """

    si = ['x', 'xd', 't', 'td']
    val = {}
    for s in si:
        apl = np.abs(df['Links'][s])._append(np.abs(df['Rechts'][s]))
        val[s] = (np.mean(apl), np.std(apl))

    return val

if __name__ == "__main__":
    path: str = "../data/s0/s0.xlsx"
    df = pd.read_excel(path, None)
    vals = calcS0(df)
    print(vals['x'])
    print(vals['xd'])
    print(vals['t'])
    print(vals['td'])

    print("\n")
    print("Sampling random from s0")
    c = np.random.choice([0, 1])
    sign = np.array([-1, -1, -1, 1]) if c == 0 else np.array([1, 1, 1, -1])
    s0 = np.array([np.random.normal(vals['x'][0], vals['x'][1]), 
                   np.random.normal(vals['xd'][0], vals['xd'][1]), 
                   np.random.normal(vals['t'][0], vals['t'][1]), 
                   np.random.normal(vals['td'][0], vals['td'][1])])
    s0 = np.multiply(sign, s0)
    print(s0)
