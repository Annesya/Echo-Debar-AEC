function energy_thrs = thresoldValue(E,thrs)
L = length(E);
energy_thrs = zeros(L,1);
for i=1:L
    if E(i,1)>=thrs
        energy_thrs(i,1) = E(i,1);
    else
      energy_thrs(i,1) = thrs;  
    end
end