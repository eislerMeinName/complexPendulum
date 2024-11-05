function PlotPendulum(u, delay, OffsetPend)
% Plot pendulum
% Input: u = [x; theta;] (cart position and pendulum angle)
%        delay: delay time in seconds by which execution of this function is delayed 
%               in order to slow down execution of simulation
%        OffsetPend: offset of pendulum angle to distinguish between
%                    inverted pendulum mode (OffsetPend = pi) and crane mode (OffsetPend = 0)

l  = 1.8;  % Length of pendulum for plotting
ll = 0.2; % Length of load
l2 = 0.3; % Distance between upper pole end to pendulum rotation axis

% Declare persistent axes handles for plotting of pendulum (hpole), cart (hcart) and load (hload)
% Note: The persistent variables can be cleared by issuing the command "clear PlotPendulum"
persistent hpole hcart hload
 
% Setup plot if one of the persistent variables is empty   
if isempty(hpole) || isempty(hcart) || isempty(hload)

    % Try to find a figure named "Pendulum"
    hpendulum = findobj('type', 'figure', 'name', 'Pendulum');
    
    if ~isempty(hpendulum)
      % Make the figure handle the current figure, if it was found
      set(0, 'CurrentFigure', hpendulum);
    else
      % If the figure does not already exist, create it
      hpendulum = figure('Name', 'Pendulum', 'NumberTitle', 'off');
      screenSize = get(0, 'ScreenSize');
      ip = get(hpendulum, 'OuterPosition');
      ip(3) = ip(3) * 1.5;
      ip(4) = ip(4) * 1.5;
      if ip(2) + ip(4) > screenSize(4)
          ip(2) = screenSize(4) - ip(4);
      end
      set(hpendulum, 'OuterPosition', ip);
    end

    % Plot the rail in blue color
    plot([-2, 2], [0 0], 'b', 'LineWidth', 2);
    
    % Limit the axes and set equal axis ratio, so that pendulum is not displayed distorted
    xlim([-2,2]);
    ylim([-2,2]);
    daspect([1,1,1]);

    % Plot cosmetics
    xlabel('x position [m]');
    ylabel('y position [m]');
    grid on;

    hold on;

    % Plot pendulum pole 
    hpole = plot([0,l2], [0, -l+l2], 'r', 'LineWidth', 2);
    
    % Plot cart as circle
    hcart = plot(0, 0, 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r');

    % Plot load
    hload = plot([0, -l+l2+ll], [0, -l+l2], 'r', 'LineWidth', 8);

    hold off;

    % Bring figure window to front
    uistack(hpendulum, 'top');
end

% Update coordinates of pendulum pole, cart and load

angle = u(2) + OffsetPend - pi;

set(hpole, 'XData', [u(1)-l2*sin(angle), u(1)+(l-l2)*sin(angle)]);
set(hpole, 'YData', [-l2*cos(angle), (l-l2)*cos(angle)]);
set(hcart, 'XData', u(1));
set(hload, 'XData', [u(1)+(l-ll-l2)*sin(angle), u(1)+(l-l2)*sin(angle)]);
set(hload, 'YData', [(l-ll-l2)*cos(angle), (l-l2)*cos(angle)]);

% Pause execution by specified amount of time in seconds
pause(delay);
