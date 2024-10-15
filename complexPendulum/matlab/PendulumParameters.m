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
% Measured

    mCart       = 0.409;           % Mass of pure cart [kg]
    mRight      = 0.1458;          % Right mass (DC rotor + pulley) [kg]
    mLeft       = 0.0805;          % Left mass (pulley) [kg]
    mBelt       = 0.045;           % Mass of belt [kg]
    mHolderP    = 0.024;           % Mass of pendulum holder [kg]
    mHolderC    = 0.045;           % Mass of cable holder (for encoder cable) [kg]

    mPole       = 0.038;           % Mass of pole [kg]
    mLoad       = 0.007;           % Mass of load [kg]

    lPole       = 0.498;           % Length of pole [m]
    rPole       = 0.00305;         % Radius of pole [m]
    lLoad       = 0.019;           % Length of load [m]
    rLoad       = 0.006;           % Radius of load [m]
    lPoleLoad   = 0.500;           % Length of pole and load [m]
    lOffset     = 0.049;           % distance upper pole end to pendulum rotation axis [m]


% ***************************************************************************************************************************
% Identified

    M           = 11.70903084;     % Control force to PWM signal ratio [N]
    fpc         = 0.034078;        % Pendulum viscuous friction dampening coefficient [1/s]
    fc          = 0.5;             % Cart viscuous friction coefficient [Ns/m]  
    Fspwm       = 0.025;           % Minimum PWM signal required to move the cart [PWM duty]


% ***************************************************************************************************************************
% Calculated

    % Total mass of cart [kg]
    mc          = mCart + mRight + mLeft + mBelt + 2*mHolderP + mHolderC

    % Total mass of pendulum [kg]
    mp          = 2 * (mPole + mLoad)

    % Combined mass of cart and pendulum [kg]
    m           = mc + mp

    % distance between center of pole mass and pendulum rotation axis [m]
    lPole0      = lPole/2 - lOffset

    % distance between center of load mass and pendulum rotation axis [m]
    lLoad0      = lPole - lLoad/2 - lOffset + (lPoleLoad - lPole)

    % Distance between center of mass of pendulum and pendulum rotation axis [m]
    l           = (2 * lPole0 * mPole + 2 * lLoad0 * mLoad) / (2 * (mPole + mLoad))

    % Moment of inertia of pole w.r.t. its center of mass [kg m^2]
    % Full cylinder
    JPole       = 1/12 * mPole * (lPole^2 + 3 * rPole^2) 

    % Moment of inertia of load w.r.t. its center of mass [kg m^2]
    % Hollow cylinder with outer radius rLoad and inner radius rPole
    JLoad       = 1/12 * mLoad * (lLoad^2 + 3 * (rLoad^2 + rPole^2))
    
    % Moment of inertia of pole and load w.r.t. pendulum rotation axis [kg m^2]
    % Account for both pendulums and shift center of mass of pole and load to rotation axis according to parallel axis theorem
    J           = 2 * (JPole + mPole * lPole0^2 + JLoad + mLoad * lLoad0^2)

    fp          = J * fpc         % Pendulum viscuous friction coefficient [Nms/rad]

    Fs          = Fspwm * M       % Static friction of cart [N], minimum force required to move the cart

    ResCartEnc  =  0.235 / 4096;   % Resolution of cart position encoder [m]
    ResPendEnc  =  2 * pi / 4096;  % Resolution of pendulum position encoder [rad]
