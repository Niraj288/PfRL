import os
import numpy as np


class Work:
        def __init__(self, seq, grid, name):
                self.seq = seq
                self.grid = grid
                self.name = name

        def minimization(self):
                text = """Stage 1 - minimisation
 &cntrl
  imin=1, maxcyc=1000, ncyc=500,
  cut=999., rgbmax=999.,igb=1, ntb=0,
  ntpr=100
 /
 """

                g = open('min1.in','w')
                g.write(text)
                g.close()

        def heating(self):
                f = open('heat1.in','w')
                f.write("""Stage 1 heating from 0 to 325K
 &cntrl
  imin=0, irest=0, ntx=1,
  nstlim=100000, dt=0.0005,
  ntc=2, ntf=2,
  ntt=1, tautp=1.0,
  tempi=0.0, temp0=325.0,
  ntpr=50, ntwx=50,
  ntb=0, igb=1,
  cut=999.,rgbmax=999.
 /
 
 """)
                f.close()

        def equillibrium(self):
                f = open('equil1.in','w')
                f.write("""Stage 2 equilibration 25000-ps
 &cntrl
  imin=0, irest=1, ntx=5,
  nstlim=1000000, dt=0.002,
  ntc=2, ntf=2,
  ntt=1, tautp=0.5,
  tempi=325.0, temp0=325.0,
  ntpr=500, ntwx=500,
  ntb=0, igb=1,
  cut=999.,rgbmax=999.
 /
 """)
                f.close()

        def gen_CA_pdb(self):
                st = 'MODEL        0\n'

                for i in range (len(self.seq)):
                        l = ['ATOM', str(i+1), 'CA', self.seq[i], 'A', str(i+1), self.grid[i][0], self.grid[i][1], self.grid[i][2], '1.00', '0.00', 'C']
                        #print ("{:>4}{:>7}{:>2} {:>5}{:>2}{:>4}{:>12}{:>8}{:>8}{:>6}{:>6}{:>12}".format(*l))
                        st += "{:>4}{:>7}{:>4} {:>4}{:>2}{:>4}{:>12}{:>8}{:>8}{:>6}{:>6}{:>12}".format(*l)+'\n'

                st += 'ENDMDL'

                g = open(self.name+'_CA.pdb','w')
                g.write(st)
                g.close()

                #os.system('java apps.BBQ -ip='+self.name+'_CA.pdb')

                #os.system('/users/nirajv/scwrl4/Scwrl4 -i '+self.name+'_CA-bb.pdb'+ ' -o '+self.name+'_gen.pdb')
                #os.system('/Users/47510753/Downloads/Scwrl4 -i '+self.name+'_CA-bb.pdb'+ ' -o '+self.name+'_gen.pdb')
                #tf = open(self.name+'_gen.pdb', 'r')
                #lines = tf.readlines()
                #tf.close()
                #os.system('reduce '+ self.name + '_gen.pdb' + ' -trim > ' + self.name + '_gen.pdb')
                #tf2 = open(self.name+'_gen.pdb', 'r')
                #lines2 = tf2.readlines()
                #tf2.close()

                #if len(lines2) < 3:
                #        g = open(self.name+'_gen.pdb', 'w')
                #        g.write(''.join(lines))
                #        g.close()

                st=''
                st+='source leaprc.protein.ff14SB\n'

                st+='pro = loadpdb '+self.name+'_CA.pdb'+'\n'

                st+='savepdb pro '+' '+self.name+'_init.pdb\n'

                st+='saveamberparm pro '+self.name+'.prmtop'+' '+self.name+'.inpcrd\n'
                st+='quit\n'

                g = open('xleap_input','w')
                g.write(st)
                g.close()

                os.system('tleap -f xleap_input')

                print ('Initial Minimization of the structure ...')
                self.minimization()
                st = 'sander -O -i min1.in -o min1.out -p name.prmtop -c name.inpcrd -r min1.rst'
                st = st.replace('name',self.name)
                os.system(st)

                print ('Heating ...')
                self.heating()
                st = 'sander -O -i heat1.in -o heat1.out -p name.prmtop -c min1.rst -r heat1.rst -x heat1.mdcrd'
                st = st.replace('name',self.name)
                os.system(st)

                print ('Equillibrium structure simulation ...')
                self.equillibrium()
                st = """sander -O -i equil1.in -p name.prmtop -c heat1.rst -r equil1.rst -o equil1.out -x equil1.mdcrd"""
                st = st.replace('name',self.name)
                os.system(st)

                print ('Saving pdb file ...')
                st = "ambpdb -p name.prmtop -c equil1.rst > name_gen.pdb"
                st = st.replace('name',self.name)
                os.system(st)





if __name__ == '__main__':
	name = '1k43'
	seq = ['ARG', 'GLY', 'LYS', 'TRP', 'THR', 'TYR', 'ASN', 'GLY', 'ILE', 'THR', 'TYR', 'GLU', 'GLY', 'ARG']
	grid = np.array([[0.0, 0.0, 0.0],
		[3.78, 0.0, 0.0],
		[7.56, 0.0, 0.0],
		[11.35, 0.0, 0.0],
		[15.14, 0.0, 0.0],
		[18.92, 0.0, 0.0],
		[22.71, 0.0, 0.0],
		[26.49, 0.0, 0.0],
		[30.28, 0.0, 0.0],
		[34.06, 0.0, 0.0],
		[37.84, 0.0, 0.0],
		[41.63, 0.0, 0.0],
		[45.42, 0.0, 0.0],
		[49.20, 0.0, 0.0]])

	w = Work(seq, grid, name)

	w.gen_CA_pdb()

# for making pdb from CA
# java apps.BBQ -bbq.fix_r12=T -ip=1k43_CA.pdb
# add side chains via SCWRL
