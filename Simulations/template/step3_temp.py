import os
import sys

#------------
# Parameters
#------------

# Output file
outfile = sys.argv[1]

# Result file
resfile = sys.argv[2]

# Flag file
flgfile = sys.argv[3]

#--------------
# Grep results
#--------------

# Read results
listfreq = ''
ivib = 0
with open(outfile, 'r') as f:
    for line in f:
        if 'ABNR>' in line:
            grms = float(line.split()[4])
        if '  VIBRATION MODE ' in line:
            if ivib>int(line.split()[2]):
                listfreq += '  {:10.8f}'.format(grms) + '\n'
            elif ivib!=0:
                listfreq += '  '
            ivib = int(line.split()[2])
            try:
                freq = '{0:=10.6f}'.format(float(line.split()[4]))
            except:
                if '*' in line:
                    freq = '{0:=10.6f}'.format(0.0)
                elif not '='==line.split()[3][-1]:
                    freq = '{0:=10.6f}'.format(
                        float(line.split()[3].split("=")[1]))
            listfreq += freq
    listfreq += '  {:10.8f}'.format(grms) + '\n'

# Write results
fres = open(resfile, 'w')
fres.write(listfreq)
fres.close()

fflg = open(flgfile, 'w')
fflg.write('FINISHED')
fflg.close()

