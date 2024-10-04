
state=[0, 0, 0, 0];

function r = relu (z)
  r = max (0, z);
endfunction

heav = @(x) (x>0) + .5*(x==0);

p1 = state * weight0' + bias0
r1 = relu(p1)
p2 = relu(r1 * weight2' + bias2)
p3 = p2*weight' + bias

tanh(p3)

p1.*heav(p1)
