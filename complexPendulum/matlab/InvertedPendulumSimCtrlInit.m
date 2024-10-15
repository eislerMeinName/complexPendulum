% ***************************************************************************************************************************
% Pendulum parameters

    PendulumParameters;     % Set pendulum parameters

	%Fs = 0;                % Uncomment to simulate model without static friction

    x0 = [0; 0; 0.15; 0]; % Initial state of pendulum [ x[m], xdot[m/s], theta[rad], thetadot[rad/s] ]

    sigmaCart = 0.0;        % Standard deviation of normally distributed measurement noise for cart encoder [m]
    sigmaPend = 0.0;        % Standard deviation of normally distributed measurement noise for pendulum encoder [rad]


% ***************************************************************************************************************************
% Parameters of state feedback controller

    % State feedback gains (as computed by pole placement or LQR)
    %K = [-0.3162   -1.9601  -22.1499   -4.1328];  % R = 1, Q = diag[ 0.1, 1, 1, 1]: Controller too slow, hits rail limits
    %K = [-1.0000   -2.6034  -24.0392   -4.4759];  % R = 1, Q = diag[   1, 1, 1, 1]: OK, but slow response
    %K = [-3.1623   -4.2796  -28.6879   -5.3254];  % R = 1, Q = diag[  10, 1, 1, 1]: OK, still a bit slow
    %K = [-10.0000   -8.5411  -39.5716   -7.3301]; % R = 1, Q = diag[ 100, 1, 1, 1]: much better
    %K = [-31.6228  -19.7494  -65.7495  -12.1855]; % R = 1, Q = diag[1000, 1, 1, 1]: now we're talking
     d = J*m - mp*mp*l*l;

 A = [[0, 1, 0, 0],
      [0, -fc*(J/d), -g*(mp*mp*l*l)/(d),(mp*l*fp)/(d)],
      [0, 0, 0, 1],
      [0, (mp*l*fc)/(d), (m*mp*l*g)/(d), (-m*fp)/(d)]]
 B = [[0],
      [J/d],
      [0],
      [(-mp*l)/(d)]]
 C = [[1, 0, 0, 0], 
     [0, 0, 1, 0]];

 Q = eye(4);
 R = 1;
 %Q = [240, 0, 0, 0; 0, 180, 0, 0; 0, 0, 12, 0; 0, 0, 0, 19]
 %R = 1;

 K = lqr(A, B, Q, R)
    %K = [-31.6228  -21.4773  -75.1344  -14.0053]; % R = 1, Q = diag([1000, 10, 200, 10]): similar, puts more emphasis on angle

    % Time constant of low pass filter for cart and pendulum velocities in state feedback controller
    Td = 0.0;


% ***************************************************************************************************************************
% Parameters of swing-up controller

    ksu = 1.3;              % Swing-up gain
    kcw = 3;                % Cart position well gain
    kvw = 3;                % Cart velocity well gain
    kem = 7;                % Energy maintenance gain
    eta = 1.05;             % Energy maintenance parameter
    E0 = 0.05;              % Desired energy for swing-up 
    xmax = 0.4;             % Distance of cart position well in both directions from rail center [m]
    vmax = 5;               % "Distance" of cart veloctiy well in both directions [m/s]
    angleSZ = 0.25;         % Limit of stabilization zone on pendulum angle for switching to state feedback controller [rad]

%% ************************************************************************
% Parameters of neural agent
load Setup1SacGain

