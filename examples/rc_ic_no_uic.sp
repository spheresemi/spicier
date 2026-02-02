RC Discharge without UIC
* .IC is applied after DC operating point calculation
V1 1 0 DC 5
R1 1 2 1k
C1 2 0 1u
.IC V(2)=2.5
.TRAN 100u 5m 0
.END
