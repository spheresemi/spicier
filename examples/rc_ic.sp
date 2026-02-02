RC Circuit with Initial Conditions
* Uses .IC to set initial capacitor voltage
V1 1 0 DC 5
R1 1 2 1k
C1 2 0 1u
.IC V(2)=2.5
.TRAN 10u 5m
.END
