RC Discharge with UIC
* Test initial conditions without DC operating point
R1 1 0 1k
C1 1 0 1u
.IC V(1)=5
.TRAN 100u 5m 0 UIC
.END
