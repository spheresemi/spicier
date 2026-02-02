RC Charging with UIC
* Using UIC to start capacitor at 0V, will charge toward 5V
* Time constant tau = R*C = 1k * 1u = 1ms
V1 1 0 5
R1 1 2 1k
C1 2 0 1u
.IC V(2)=0
.PRINT TRAN V(2)
.TRAN 0.5m 5m UIC
.END
