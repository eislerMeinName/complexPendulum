%
% Pendulum parameters
% 
%
% ***************************************************************************************************************************
% Constants

    g           = 9.81;            % Gravity [m/s^2]
    Ts          = 0.01;            % Sampling time [s]
    PlotDelay   = Ts - 0.001;      % Delay time after plotting for slowing down the simulation approx. to actual time
    OffsetCart  = 0;               % Cart position offset [m]
    OffsetPend  = pi;              % Pendulum position offset [rad]


% ***************************************************************************************************************************
    m = 0.82216
    mc = 0.4512
    mp = 0.096739999999999999
    l = 0.2545253256150507
    J = 0.0087878516209625
    M1 = 11.166666666666666666
    M0 = -0.2222222
    ep = 0.0002196962905240625
    ec = 0.8205818364648804
    muc = 0.0657069625931102
    mus = 0.500372779995686

    ResCartEnc  =  0.235 / 4096;   % Resolution of cart position encoder [m]
    ResPendEnc  =  2 * pi / 4096;  % Resolution of pendulum position encoder [rad]
