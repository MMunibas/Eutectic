//18.2.2020  Taken over from CoupleSurface, and cleaned up: removed all options METHOD and DIMER, as well as pseudo-MD
//28.1.2021  Some weird allocation error; strange interference between calculation with sporadic norm errors when more than one job is running.
//           suspect f4tensor to be the problem, even though used it many times before and never has been a problem, but always felt uneasy 
//           about it. Test f4tensor.txt insted (http://numerical.recipes/forum/showthread.php?t=450), which has clearer structure


#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

/*Simulation Parameters*/
#define nprocmax          %NPRC%
#define anharmonicity     27.          //anharmonicity of an isolated C=O vibration
#define dw_isotope        75.          //isotope shift 13C14N; anharmonicity and isotope shift as in J. Phys. Chem. B, 110, 2006, 19990
#define dt                0.2          //timestep for Fouriertransforms; time step of MD, currently, 100fs
#define nt                16           //number of time steps during coherence times; has to be a number 2^n
#define nt2               100           //number of time steps during population time;
#define dnt2              5            //number of skiped steps during population time
#define wavg              %WAVG%       //average center frequency
#define woff              50           //offset frequency to center spectra as good as possible in window; 
#define dipole            0.073        //transition dipole moment in units of e*A; according to J. Phys. Chem. A 2014, 118, 2463, it is 
                                       //0.3-0.4D, the average of which corresponds to 0.07 e*A.
#define ntstep            %NTSP%       //step with which it loops through trajectory

/*stuff related to input file*/
#define nSCN              75           //number of SCN molecules 
#define ntMD              %NTMD%

/*other constants*/
#define PI                3.141592653589793
#define cw                0.188       //transformation factor frequency from cm-1 into ps-1
#define epsilon           116140.           //transformation factor energy from e*e/A to cm-1: e*e/4/PI/e0*1e10/h/c

typedef struct COMPLEX {float re,im;} complex;
//typedef struct DCOMPLEX {double re,im;} dcomplex;

/* Numerical Recipes Routines */

void nrerror(char error_text[]);
double *dvector(long nl, long nh);
float *vector(long nl, long nh);
float **matrix(long nrl, long nrh, long ncl, long nch);
void free_matrix(float **m, long nrl, long nrh, long ncl, long nch);
void free_complexmatrix(complex **m, long nrl, long nrh, long ncl, long nch);
void free_vector(float *v, long nl, long nh);
void free_imatrix(int **m,long nrl,long nrh,long ncl,long nch);
void free_f3tensor(float ***t, long nrl, long nrh, long ncl, long nch,
        long ndl, long ndh);
float ***f3tensor(long nrl, long nrh, long ncl, long nch, long ndl, long ndh);
int **imatrix(long nrl, long nrh, long ncl, long nch);
int *ivector(long nl, long nh);
void free_ivector(int *v, long nl, long nh);
complex *complexvector(long nl, long nh);
void free_complexvector(complex *v, long nl, long nh);
complex **complexmatrix(long nrl, long nrh, long ncl, long nch);
complex ***complex3tensor(long nrl, long nrh, long ncl, long nch, long ndl, long ndh);
complex ****complex4tensor(long nrl, long nrh, long ncl, long nch, long ndl, long ndh, long nel, long neh);
void four1(float data[], unsigned long nn, int isign);
void eigsrt(float d[], float **v, int n);
float gasdev(long *idum);
void tred2(float **a, int n, float d[], float e[]);
void tqli(float d[], float e[], int n, float **z);
float ran1(long *idum);
void bessjy(double x, double xnu, double *rj, double *ry, double *rjp, double *ryp);

/*My Routines */
void fft(complex *P, int n);
void fft2D(complex **R, int n);
void write_spectra(complex *S,complex ***Rr,complex ***Rnr);
void CalcProp(complex **U,float **H,int n);
void Propagate(complex *psi2,complex **U,complex *psi1,int n);
void CalcMu12(float **mu12, float *mu01,int n);
void mult_H(complex *psi1,complex *psi0, float *H,int *iH2, int *jH2,int n2,int nH2);
int two_exciton_matrix(float **H2,int *iH2,int *jH2,float ***H1,int n);
void Propagate2(complex *psi2,float *H,int *iH2,int *jH2,complex *psi1,int n2,int nH2);
void two_exciton_matrix_full(float **H2,float **H1,int n);
void DFT(complex *b,complex *a, int n);

//random generator
long idum=-1;

//for Chebychev scheme
#define maxiter 50
double R;
double a[maxiter];
double w0;
int niter,mix,run,sample;

int main(int iarg, char **input)
{
  FILE *out,*trajectory;
  complex *R1,***Rr,***Rnr,*S1,***Sr,***Snr,***U,**psi1,***psi2,****psi3,****psiE23,****psiE13,ctmp,r12,r13,r14,r23,r24,r34;
  int n2,i,j,k,it,it1,it2,it3,itmp,*iH2,*jH2,nH2,nproc;
  float ***H0,***H,**H2,**mu01,***mu12,*wshift,tmp,tmp1,tmp2,tmp3,time1=0,time2=0,time3=0,time4=0,time5=0,norm0,**norm1,**norm2,*rbox,rboxavg=0,couplemax=0;
  float normdiff,normdiffmax=0,***x,***mu,r[4],Vmax,Vmin,dr,f,tilt,t,ftmp,xN[4],xC[4],wmin,wmax;
  double dtmp1,dtmp2,dtmp3,dw;
  int isample,n,itMD,ifile,natom,iatom,iSCN,itmin,itmax,iPol;
  clock_t timer;
  char inbuff[500],name[50],*chartmp;

  itmp=sscanf(input[1],"%d",&mix);
  itmp=sscanf(input[2],"%d",&run);
  itmp=sscanf(input[3],"%d",&sample);
  printf("Mixing ratio: %d\n",mix);
  printf("Run: %d\n",run);
  printf("Sample: %d\n\n",sample);

  nproc=1;
#ifdef _OPENMP
  nproc=omp_get_max_threads();
  if(nproc>nprocmax) nproc=nprocmax;
#endif  
  printf("Will be running on %d processors\n\n",nproc);

  Vmax=2*2*2*PI/2/dt/cw;     //determined as twice the borders of the frequency space (needed for doubly excited states); times a factor 2 as reserve
  Vmin=-2*2*2*PI/2/dt/cw;
  R=(Vmax-Vmin)/2*dt*cw;
  for(niter=0;niter<maxiter;niter++) 
    {
    bessjy(R,(double)niter,&a[niter],&dtmp1,&dtmp2,&dtmp3);
    if(niter>=1) a[niter]*=2;
    printf("%d: %13.11f\n",niter,a[niter]);
    if(fabs(a[niter])<1e-10) break;
    }
  niter--;
  printf("Prepare for Chebychev Iteration: Vmin=%5.1f,  Vmax=%5.1f, niter=%d\n",Vmin,Vmax,niter);

  n=nSCN;
  printf("\nNumber of SCN molecules: %d\n",n);

//allocate stuff
  x=f3tensor(1,ntMD,1,n,1,3);  
  mu=f3tensor(1,ntMD,1,n,1,3);
  H0=f3tensor(1,ntMD,1,n,1,n);
  rbox=vector(1,ntMD);
  n2=n*(n+1)/2;
  H=f3tensor(1,2*nt+nt2*dnt2,1,n,1,n);
  U=complex3tensor(1,2*nt+nt2*dnt2,1,n,1,n);
  mu01=matrix(0,2*nt+nt2*dnt2,1,n);
  mu12=f3tensor(0,2*nt+nt2*dnt2,1,n2,1,n);
  wshift=vector(1,n);

  psi1=complexmatrix(0,2*nt+nt2*dnt2,1,n);
  psi2=complex3tensor(0,nt,0,nt+nt2*dnt2,1,n);
  psi3=complex4tensor(0,nt,0,nt2,0,nt,1,n);
  psiE13=complex4tensor(0,nt,0,nt2,0,nt,1,n2);
  psiE23=complex4tensor(0,nt,0,nt2,0,nt,1,n2);
 
  iH2=ivector(1,n*n*n);
  jH2=ivector(1,n*n*n);
  H2=matrix(1,2*nt+nt2*dnt2,1,n*n*n);    //sparse matrix representation of H2
  R1=complexvector(0,nt);  
  S1=complexvector(0,2*nt); //twice as long vector needed for zero padding
  for(it1=0;it1<=nt;it1++) {R1[it1].re=0;R1[it1].im=0;}   
  Sr=complex3tensor(0,nt2,0,2*nt,0,2*nt); 
  Rr=complex3tensor(0,nt2,0,nt,0,nt);  
  for(it2=0;it2<=nt2;it2++) for(it1=0;it1<=nt;it1++) for(it3=0;it3<=nt;it3++) {Rr[it2][it3][it1].re=0;Rr[it2][it3][it1].im=0;}
  Rnr=complex3tensor(0,nt2,0,nt,0,nt); 
  Snr=complex3tensor(0,nt2,0,2*nt,0,2*nt); 
  for(it2=0;it2<=nt2;it2++) for(it1=0;it1<=nt;it1++) for(it3=0;it3<=nt;it3++) {Rnr[it2][it3][it1].re=0;Rnr[it2][it3][it1].im=0;}
  norm1=matrix(0,nt2,0,nt);
  norm2=matrix(0,nt2,0,nt);

  isample=0;
  //  for(ifile=1;ifile<=nfile;ifile++)
    {
//read MD trajectory
    sprintf(name,"snaps_300_%d_%d_%d_SCN.xyz",mix,run,sample);
    if((trajectory=fopen(name,"rt"))==NULL) {printf("Error opening file %s\n",name);exit(1);}
    printf("\nRead from file: %s\n",name);
    for(it=1;it<=ntMD;it++)
      {
      itmp=fscanf(trajectory,"%d\n",&natom);
      if(it==1) printf("Number of atoms: %d\n",natom);
      //itmp=fscanf(trajectory,"%s %s %f %s",inbuff,inbuff,&t,inbuff); 
      chartmp=fgets(inbuff,500,trajectory); 
      //     printf("Time point %4d:   %s\n",it,inbuff);
      iSCN=0;
      for(iatom=1;iatom<=natom;iatom++)
	{
        itmp=fscanf(trajectory,"%s %f %f %f",inbuff,&xN[1],&xN[2],&xN[3]);
        if(strcmp(inbuff,"N3C")==0)
	  {
	  iSCN++;
	  iatom++;
          itmp=fscanf(trajectory,"%s %f %f %f",inbuff,&xC[1],&xC[2],&xC[3]);
	  if(strcmp(inbuff,"C3N")!=0) {printf("Wrong atom type %s != C3N\n",inbuff); exit(1);}
	  for(j=1;j<=3;j++) x[it][iSCN][j]=(xN[j]+xC[j])/2;
          for(j=1;j<=3;j++) mu[it][iSCN][j]=(xN[j]-xC[j]);
          for(j=1,norm0=0;j<=3;j++) norm0+=mu[it][iSCN][j]*mu[it][iSCN][j];
	  if(norm0>2) printf("CN bond length %4.2f\n",sqrt(norm0));
          for(j=1;j<=3;j++) mu[it][iSCN][j]*=dipole/sqrt(norm0);
	  }
	}
      if(iSCN!=nSCN) {printf("Number of read SCN molecules %d != %d\n",iSCN,nSCN); exit(1);}
      }
    fclose(trajectory);

//read box sizes    
   sprintf(name,"ucell_300_%d_%d_%d_SCN.dat",mix,run,sample);
   if((trajectory=fopen(name,"rt"))==NULL) {printf("Error opening file %s\n",name);exit(1);}
   printf("\nRead from file: %s\n",name);
   for(it=1;it<=ntMD;it++)
      {
      itmp=fscanf(trajectory,"%s %f %s %s",inbuff,&rbox[it],inbuff,inbuff); 
      rboxavg+=rbox[it]/ntMD;
      }
   printf("Average box size %5.2f\n",rboxavg);

//calculate time series of dipole-dipole couplings
   couplemax=0;
   for(it=1;it<=ntMD;it++) 
     {
     for(i=2;i<=n;i++) for(j=1;j<i;j++) 
       {
       for(k=1;k<=3;k++) r[k]=x[it][i][k]-x[it][j][k]-floor((x[it][i][k]-x[it][j][k])/rbox[it]+0.5)*rbox[it];
       for(k=1,r[0]=0;k<=3;k++) r[0]+=r[k]*r[k]; 
       for(k=1,tmp1=0,tmp2=0,tmp3=0;k<=3;k++) {tmp1+=mu[it][i][k]*mu[it][j][k];tmp2+=mu[it][i][k]*r[k];tmp3+=mu[it][j][k]*r[k];}
       tmp=tmp1/(r[0]*sqrt(r[0]))-3*tmp2*tmp3/(r[0]*r[0]*sqrt(r[0]));
       H0[it][i][j]=epsilon*tmp; 
       H0[it][j][i]=H0[it][i][j];
       if(fabs(H0[it][i][j])>fabs(couplemax)) couplemax=H0[it][i][j];
       }
     }
   printf("Largest coupling %5.2f cm-1\n",couplemax);

//read instantaneous frequencies  
    sprintf(name,"freqs_300_%d_%d_%d_SCN.dat",mix,run,sample);
    if((trajectory=fopen(name,"rt"))==NULL) {printf("Error opening file %s\n",name);exit(1);}
    printf("\nRead from file: %s\n",name);
    w0=0; wmin=10000; wmax=0;
    for(it=1;it<=ntMD;it++)
      {
      itmp=fscanf(trajectory,"%f",&ftmp);
      //     printf("Time point: %5.0f\n",ftmp);
      for(i=1;i<=nSCN;i++) 
        {
        itmp=fscanf(trajectory,"%f",&H0[it][i][i]);
        w0+=H0[it][i][i];
	if(wmin>H0[it][i][i]) {wmin=H0[it][i][i];itmin=it;}
        if(wmax<H0[it][i][i]) {wmax=H0[it][i][i];itmax=it;}
        }
      }
    fclose(trajectory);
    w0=wavg;
    dw=0;
    for(it=1;it<=ntMD;it++) for(i=1;i<=nSCN;i++) dw+=(H0[it][i][i]-w0)*(H0[it][i][i]-w0);
    dw=sqrt(dw/ntMD/nSCN);
    printf("Average frequency: %6.1f cm-1\nStd. Dev.:         %6.1f cm-1\nLowest frequency:  %6.1f cm-1 at time point %d\nLargest frequency: %6.1f cm-1 at time point %d\n\n",w0,dw,wmin,itmin,wmax,itmax);

//print two example Hamiltonians early and late time
    printf("Hamiltonian, time: 0 ps\n");
    for(i=1;i<=20;i++) {for(j=1;j<=20;j++) printf(" %5.1f ",H0[1][i][j]);printf("\n");} printf("\n");
    printf("Hamiltonian, time: %6.1f ps\n",dt*ntMD);
    for(i=1;i<=20;i++) {for(j=1;j<=20;j++) printf(" %5.1f ",H0[ntMD][i][j]);printf("\n");} printf("\n");

//loop through trajectory with different starting points
    // printf("\nMinimum # time steps needed %d\n\n",2*nt+nt2*dnt2); 
    for(itMD=1;itMD<=ntMD-2*nt-nt2*dnt2;itMD+=ntstep)
      {
      normdiff=0.;
      timer=clock();
      iPol=(isample%3)+1;
      printf("isample: %4d (iPol %d) at starting time %7.1f\n",isample+1,iPol,dt*itMD);
      for(i=1;i<=n;i++) 
          if(ran1(&idum)<0.5) wshift[i]=0;
          else                wshift[i]=-dw_isotope;
//perform sliding average over H0, and add isotopes 
      for(it=1;it<2*nt+nt2*dnt2;it++)
        {
        for(i=1;i<=n;i++) for(j=1;j<=n;j++) H[it][i][j]=H0[itMD+it-1][i][j];
        for(i=1;i<=n;i++) H[it][i][i]=H0[itMD+it-1][i][i]-w0+woff+wshift[i];
        }
//print example Hamiltonian 
//      for(i=1;i<=20;i++) {for(j=1;j<=20;j++) printf(" %5.1f ",H[1][i][j]);printf("\n");} printf("\n");
 
      nH2=two_exciton_matrix(H2,iH2,jH2,H,n);
      if(nH2>n*n*n) {printf("Allocation for the two-exciton matrix to small; size needs to be: %d\n",nH2);exit(1);}

 //extract corresponding dipole moments
       for(it=0;it<2*nt+nt2*dnt2;it++) for(i=1;i<=n;i++) mu01[it][i]=mu[itMD+it][i][iPol]/sqrt(n)/dipole; 
      //for(it=0;it<2*nt+nt2*dnt2;it++) for(i=1;i<=n;i++) mu01[it][i]=(fabs(mu[itMD+it][i][1])+fabs(mu[itMD+it][i][2])+fabs(mu[itMD+it][i][3]))/3.0/sqrt(n)/dipole; 
      for(it=0;it<2*nt+nt2*dnt2;it++) CalcMu12(mu12[it],mu01[it],n);

      time1+=(float)(clock()-timer)/CLOCKS_PER_SEC;
      timer=clock();
#pragma omp parallel for num_threads(nproc) default(none) shared(U,H,n)  private(it) 
      for(it=1;it<2*nt+nt2*dnt2;it++) CalcProp(U[it],H[it],n); 
      time1+=(float)(clock()-timer)/CLOCKS_PER_SEC/nproc;


//calculate time series of wavefunctions from various starting points
      timer=clock();
      for(i=1;i<=n;i++) {psi1[0][i].re=mu01[0][i];psi1[0][i].im=0;}
      for(it=1;it<2*nt+nt2*dnt2;it++) Propagate(psi1[it],U[it],psi1[it-1],n);
      for(it1=0;it1<nt;it1++)
        {
        for(i=1;i<=n;i++) {psi2[it1][0][i].re=mu01[it1][i];psi2[it1][0][i].im=0;}
        for(it=1;it<nt+nt2*dnt2;it++) Propagate(psi2[it1][it],U[it1+it],psi2[it1][it-1],n);
        }
      for(it1=0;it1<nt;it1++) for(it2=0;it2<nt2;it2++) 
        {
        for(i=1;i<=n;i++) {psi3[it1][it2][0][i].re=mu01[it1+it2*dnt2][i];psi3[it1][it2][0][i].im=0;}
        for(it=1;it<nt;it++) Propagate(psi3[it1][it2][it],U[it1+it2*dnt2+it],psi3[it1][it2][it-1],n);
        }
      time2+=(float)(clock()-timer)/CLOCKS_PER_SEC;

//calculate time series of wavefunctions related to excited state absorption
      timer=clock();
#pragma omp parallel for num_threads(nproc) default(none) shared(psiE13,psiE23,mu12,psi1,psi2,H2,iH2,jH2,norm1,norm2,n2,nH2,n)  private(it1,it2,i,j,it,tmp) 
      for(it1=0;it1<nt;it1++) for(it2=0;it2<nt2;it2++) 
        {
        for(i=1;i<=n2;i++) 
	  {
          psiE13[it1][it2][0][i].re=0;psiE13[it1][it2][0][i].im=0;
          psiE23[it1][it2][0][i].re=0;psiE23[it1][it2][0][i].im=0;
          for(j=1;j<=n;j++) 
	    {
	    psiE13[it1][it2][0][i].re+=mu12[it1+it2*dnt2][i][j]*psi1[it1+it2*dnt2][j].re;
            psiE13[it1][it2][0][i].im+=mu12[it1+it2*dnt2][i][j]*psi1[it1+it2*dnt2][j].im;
            psiE23[it1][it2][0][i].re+=mu12[it1+it2*dnt2][i][j]*psi2[it1][it2*dnt2][j].re;
            psiE23[it1][it2][0][i].im+=mu12[it1+it2*dnt2][i][j]*psi2[it1][it2*dnt2][j].im;
	    }
	  }
        for(i=1,norm1[it1][it2]=0;i<=n2;i++) norm1[it1][it2]+=psiE13[it1][it2][0][i].re*psiE13[it1][it2][0][i].re+psiE13[it1][it2][0][i].im*psiE13[it1][it2][0][i].im;
        for(i=1,norm2[it1][it2]=0;i<=n2;i++) norm2[it1][it2]+=psiE23[it1][it2][0][i].re*psiE23[it1][it2][0][i].re+psiE23[it1][it2][0][i].im*psiE23[it1][it2][0][i].im;
        for(it=1;it<nt;it++) 
	  {
	  Propagate2(psiE13[it1][it2][it],H2[it1+it2*dnt2+it],iH2,jH2,psiE13[it1][it2][it-1],n2,nH2);
	  Propagate2(psiE23[it1][it2][it],H2[it1+it2*dnt2+it],iH2,jH2,psiE23[it1][it2][it-1],n2,nH2);
	  }
        for(i=1,tmp=0;i<=n2;i++) tmp+=psiE13[it1][it2][nt-1][i].re*psiE13[it1][it2][nt-1][i].re+psiE13[it1][it2][nt-1][i].im*psiE13[it1][it2][nt-1][i].im;
        norm1[it1][it2]=fabs(sqrt(tmp/norm1[it1][it2])-1);
      //printf("norm1: %9.7f %9.7f\n",norm1[it1][it2],sqrt(tmp));
        for(i=1,tmp=0;i<=n2;i++) tmp+=psiE23[it1][it2][nt-1][i].re*psiE23[it1][it2][nt-1][i].re+psiE23[it1][it2][nt-1][i].im*psiE23[it1][it2][nt-1][i].im;
        norm2[it1][it2]=fabs(sqrt(tmp/norm2[it1][it2])-1);
        }
      for(it1=0;it1<nt;it1++) for(it2=0;it2<nt2;it2++)  
        {
        if(norm1[it1][it2]>normdiff) normdiff=norm1[it1][it2];
        if(norm2[it1][it2]>normdiff) normdiff=norm2[it1][it2];
        }
      time3+=(float)(clock()-timer)/CLOCKS_PER_SEC/nproc;

      if(normdiff>normdiffmax) normdiffmax=normdiff;
      if(normdiff>.0001) printf("Norm diverged too much: %10.7f\n",normdiff);
      else  //consider only if normdiff is not too large
        {
	isample++;     
        timer=clock();
//calculate linear response
        for(it=0;it<nt;it++) for(i=1;i<=n;i++) {R1[it].re+=mu01[it][i]*psi1[it][i].re;R1[it].im+=mu01[it][i]*psi1[it][i].im;}

//calculate 2D responses
#pragma omp parallel for num_threads(nproc) default(none) shared(psi1,psi2,psi3,psiE13,psiE23,Rnr,Rr,mu12,mu01,n,n2)  private(it1,it2,it3,i,j,r12,r13,r23,r14,r24,r34,ctmp) 
        for(it2=0;it2<=nt2;it2++) for(it1=0;it1<nt;it1++)
          {
          for(i=1,r12.re=0,r12.im=0;i<=n;i++) {r12.re+=mu01[it1][i]*psi1[it1][i].re;r12.im+=mu01[it1][i]*psi1[it1][i].im;}
          for(i=1,r13.re=0,r13.im=0;i<=n;i++) {r13.re+=mu01[it1+it2*dnt2][i]*psi1[it1+it2*dnt2][i].re;  r13.im+=mu01[it1+it2*dnt2][i]*psi1[it1+it2*dnt2][i].im;}
          for(i=1,r23.re=0,r23.im=0;i<=n;i++) {r23.re+=mu01[it1+it2*dnt2][i]*psi2[it1][it2*dnt2][i].re; r23.im+=mu01[it1+it2*dnt2][i]*psi2[it1][it2*dnt2][i].im;}
          for(it3=0;it3<nt;it3++)
	    {
            for(i=1,r14.re=0,r14.im=0;i<=n;i++) {r14.re+=mu01[it1+it2*dnt2+it3][i]*psi1[it1+it2*dnt2+it3][i].re;  r14.im+=mu01[it1+it2*dnt2+it3][i]*psi1[it1+it2*dnt2+it3][i].im;}
            for(i=1,r24.re=0,r24.im=0;i<=n;i++) {r24.re+=mu01[it1+it2*dnt2+it3][i]*psi2[it1][it2*dnt2+it3][i].re; r24.im+=mu01[it1+it2*dnt2+it3][i]*psi2[it1][it2*dnt2+it3][i].im;}
            for(i=1,r34.re=0,r34.im=0;i<=n;i++) {r34.re+=mu01[it1+it2*dnt2+it3][i]*psi3[it1][it2][it3][i].re;     r34.im+=mu01[it1+it2*dnt2+it3][i]*psi3[it1][it2][it3][i].im;}
//GB response functions, non rephasing
            Rnr[it2][it3][it1].re-=r12.re*r34.re-r12.im*r34.im;
            Rnr[it2][it3][it1].im-=r12.re*r34.im+r12.im*r34.re;
//GB response functions, rephasing
            Rr[it2][it3][it1].re -=r12.re*r34.re+r12.im*r34.im;
            Rr[it2][it3][it1].im -=r12.re*r34.im-r12.im*r34.re;
//SE response function, nonrephasing 
            Rnr[it2][it3][it1].re-=r23.re*r14.re+r23.im*r14.im;
            Rnr[it2][it3][it1].im-=r23.re*r14.im-r23.im*r14.re;
//SE response function, rephasing 
            Rr[it2][it3][it1].re -=r13.re*r24.re+r13.im*r24.im;
            Rr[it2][it3][it1].im -=r13.re*r24.im-r13.im*r24.re;
            for(i=1;i<=n;i++) 
	      {
//EA resonse function, nonrepahsing
              for(j=1,ctmp.re=0,ctmp.im=0;j<=n2;j++) {ctmp.re+=psiE13[it1][it2][it3][j].re*mu12[it1+it2*dnt2+it3][j][i];ctmp.im+=psiE13[it1][it2][it3][j].im*mu12[it1+it2*dnt2+it3][j][i];}
	      Rnr[it2][it3][it1].re+=ctmp.re*psi2[it1][it2*dnt2+it3][i].re+ctmp.im*psi2[it1][it2*dnt2+it3][i].im;
	      Rnr[it2][it3][it1].im+=ctmp.im*psi2[it1][it2*dnt2+it3][i].re-ctmp.re*psi2[it1][it2*dnt2+it3][i].im;
//EA resonse function, rephasing
              for(j=1,ctmp.re=0,ctmp.im=0;j<=n2;j++) {ctmp.re+=psiE23[it1][it2][it3][j].re*mu12[it1+it2*dnt2+it3][j][i];ctmp.im+=psiE23[it1][it2][it3][j].im*mu12[it1+it2*dnt2+it3][j][i];}
	      Rr[it2][it3][it1].re +=ctmp.re*psi1[it1+it2*dnt2+it3][i].re+ctmp.im*psi1[it1+it2*dnt2+it3][i].im;
	      Rr[it2][it3][it1].im +=ctmp.im*psi1[it1+it2*dnt2+it3][i].re-ctmp.re*psi1[it1+it2*dnt2+it3][i].im;
	      }
  	    }
          }
        time4+=(float)(clock()-timer)/CLOCKS_PER_SEC/nproc;
	}

//write results; do for every iteration step
      timer=clock();
      for(it=0;it<nt;it++) {S1[it].re=R1[it].re/isample;S1[it].im=R1[it].im/isample;}

      for(it2=0;it2<nt2;it2++) for(it1=0;it1<nt;it1++) for(it3=0;it3<nt;it3++)
        {
        Snr[it2][it3][it1].re=Rnr[it2][it3][it1].re/isample;
        Snr[it2][it3][it1].im=Rnr[it2][it3][it1].im/isample;
        Sr[it2][it3][it1].re =Rr[it2][it3][it1].re/isample;
        Sr[it2][it3][it1].im =Rr[it2][it3][it1].im/isample;
        }

      sprintf(name,"time_lin_%d.dat",mix);
      out=fopen(name,"wt");
      for (it1=0;it1<nt;it1++) fprintf(out,"%6.2f %8.4f \n",dt*it1,S1[it1].re);
      fclose(out);

      fft(S1,nt);
      for(it2=0;it2<=nt2;it2++)
        {
        fft2D(Sr[it2],nt);
        fft2D(Snr[it2],nt);
        }
      write_spectra(S1,Sr,Snr);

      time5+=(float)(clock()-timer)/CLOCKS_PER_SEC;
      fflush(stdout);
      }
    }
    

  printf("Maximum deviation of norm during propagation of doubly excited states: %9.7f\n\n",normdiffmax);
  printf("\nTiming:\n");
  tmp=time1+time2+time3+time4+time5;
  printf("Total time:                      %7.2fs  \n",tmp); 
  printf("Calculate propagation matrices:  %7.2fs  %4.1f%%\n",time1,time1/tmp*100);
  printf("Propagate singly excited states: %7.2fs  %4.1f%%\n",time2,time2/tmp*100);
  printf("Propagate doubly excited states: %7.2fs  %4.1f%%\n",time3,time3/tmp*100);
  printf("Calculate response functions:    %7.2fs  %4.1f%%\n",time4,time4/tmp*100);
  printf("Fourier transforms and writing:  %7.2fs  %4.1f%%\n",time5,time5/tmp*100);
 
}
/******************************************************************************/
void CalcMu12(float **mu12, float *mu01,int n)
{
  int i,j,k,n2,i1;
    
  n2=n*(n+1)/2;
  for (i=1,i1=1;i<=n;i++) for(j=i;j<=n;j++,i1++) for(k=1;k<=n;k++)
    {
    if(i==j&&i==k) mu12[i1][k]=sqrt(2)*mu01[k];
    if(i!=j&&i==k) mu12[i1][k]=mu01[j];
    if(i!=j&&j==k) mu12[i1][k]=mu01[i];
    }

  //for(i=1;i<=n2;i++) {for(j=1;j<=n;j++) printf("%6.3f ",mu12[i][j]);printf("\n");}

}
/*******************************************************************************/
void Propagate2(complex *psi2,float *H,int *iH2,int *jH2,complex *psi1,int n2,int nH2)
{
 int m,k,i0=0,i1=1,i2=2;
 int i,j,it,itmp,iiter;
 long is;
 double dtmp1,step,dtmp2,dtmp3;
 float tmp1,tmp2,tmp3,norm,norm2,normold;
 complex **phi;


  step=dt*cw;
  phi=complexmatrix(0,2,1,n2);
//initialize iteration
  for(i=1;i<=n2;i++) 
     {
     phi[i0][i].re=psi1[i].re;
     phi[i0][i].im=psi1[i].im;
     psi2[i].re=a[0]*psi1[i].re;  
     psi2[i].im=a[0]*psi1[i].im;
     }
  mult_H(phi[i1],phi[i0],H,iH2,jH2,n2,nH2);
  dtmp1=step/R;
  for(i=1;i<=n2;i++) 
     {
     dtmp2=phi[i1][i].re;
     phi[i1][i].re=+phi[i1][i].im*dtmp1;
     phi[i1][i].im=-dtmp2*dtmp1;
     psi2[i].re+=phi[i1][i].re*a[1];
     psi2[i].im+=phi[i1][i].im*a[1];
     }

//recurance relation
  for(iiter=2;iiter<=niter;iiter++)
    {
    mult_H(phi[i2],phi[i1],H,iH2,jH2,n2,nH2);
    dtmp1=2*step/R;
    for(i=1;i<=n2;i++) 
      {
      dtmp2=phi[i2][i].re;
      phi[i2][i].re=+dtmp1*phi[i2][i].im+phi[i0][i].re;
      phi[i2][i].im=-dtmp1*dtmp2        +phi[i0][i].im;
      psi2[i].re+=phi[i2][i].re*a[iiter];
      psi2[i].im+=phi[i2][i].im*a[iiter];
      }
    itmp=i0;i0=i1;i1=i2;i2=itmp;
    } 
  free_complexmatrix(phi,0,2,1,n2);
}
/*********************************************************************************************/
void mult_H(complex *psi1,complex *psi0, float *H,int *iH2, int *jH2,int n2,int nH2)
{

  int i,j;

  for(i=1;i<=n2;i++) {psi1[i].re=0;psi1[i].im=0;}
  for(i=1;i<=nH2;i++) {psi1[iH2[i]].re+=H[i]*psi0[jH2[i]].re;psi1[iH2[i]].im+=H[i]*psi0[jH2[i]].im;}  
}

/*******************************************************/
void Propagate(complex *psi2,complex **U,complex *psi1,int n)
//used with METHOD 0
{
  int i,j;
  for(i=1;i<=n;i++) for(j=1,psi2[i].re=0,psi2[i].im=0;j<=n;j++) 
    {
    psi2[i].re+=U[i][j].re*psi1[j].re-U[i][j].im*psi1[j].im;
    psi2[i].im+=U[i][j].re*psi1[j].im+U[i][j].im*psi1[j].re;
    }
}
/******************************************************************************/
void CalcProp(complex **U,float **H,int n)
{
//Calculate time-propagation matrix U=Exp(-i H dt)
  int i,j,k;
  float *D,*e,**C;
  /*
  static int firstmark=0;

  if(firstmark==0)
    {
#if METHOD==1
    n2=n;
#else
    n2=n*(n+1)/2;   //allocate space also for doubly excited states
#endif
    firstmark=1;
    D=vector(1,n2);
    e=vector(1,n2);
    C=matrix(1,n2,1,n2);
    }
  */


  D=vector(1,n);
  e=vector(1,n);
  C=matrix(1,n,1,n);
  for(i=1;i<=n;i++) for(j=1;j<=n;j++) C[i][j]=H[i][j];
  tred2(C,n,D,e);
  tqli(D,e,n,C);
  for(i=1;i<=n;i++) for(k=1;k<=n;k++) for(j=1,U[i][k].re=0,U[i][k].im=0;j<=n;j++) 
    {
    U[i][k].re+=C[i][j]*cos(-D[j]*cw*dt)*C[k][j];
    U[i][k].im+=C[i][j]*sin(-D[j]*cw*dt)*C[k][j];
    }
  free_vector(D,1,n);
  free_vector(e,1,n);
  free_matrix(C,1,n,1,n);
  //  for(i=1;i<=n;i++) {for(j=1;j<=n;j++) printf(" %7.3f ",U[i][j].re);printf("\n");} printf("\n");
  //for(i=1;i<=n;i++) {for(j=1;j<=n;j++) printf(" %7.3f ",U[i][j].im);printf("\n");} printf("\n\n");
}

/******************************************************************************/
void two_exciton_matrix_full(float **H2,float **H1,int n)
{
int i,j,k,l,i1,i2;

i1=0;i2=0;

for (i=1;i<=n;i++)
  for (j=i;j<=n;j++)
    {
    i1++;
    i2=0;
    for (k=1;k<=n;k++)
     for (l=k;l<=n;l++)
       {
       i2++;
       H2[i1][i2]=0;
       if ((i==k)&&(j==l)&&(i==j)) H2[i1][i2]=2*H1[i][i]-anharmonicity;    /* <2i|V|2i> */

       if ((i==k)&&(j==l)&&(i!=j)) H2[i1][i2]=H1[i][i]+H1[j][j];           /* <1i1j|V|1i1j> */

       if ((i==j)&&(j==l)&&(i!=k)) H2[i1][i2]=sqrt(2)*H1[i][k];            /* <2i|V|1i1j>  */
       if ((i==j)&&(i==k)&&(j!=l)) H2[i1][i2]=sqrt(2)*H1[i][l];
       if ((k==l)&&(i==k)&&(j!=l)) H2[i1][i2]=sqrt(2)*H1[j][l];
       if ((k==l)&&(j==l)&&(i!=k)) H2[i1][i2]=sqrt(2)*H1[i][k];

       if ((i==k)&&(j!=l)&&(i!=j)&&(k!=l)) H2[i1][i2]=H1[j][l];             /* <ij|V|kl>  */
       if ((j==l)&&(i!=k)&&(i!=j)&&(k!=l)) H2[i1][i2]=H1[i][k];
       if ((i==l)&&(j!=k)&&(i!=j)&&(k!=l)) H2[i1][i2]=H1[j][k];
       if ((j==k)&&(i!=l)&&(i!=j)&&(k!=l)) H2[i1][i2]=H1[i][l];
       }
    }
}
/******************************************************************************/
int two_exciton_matrix(float **H2,int *iH2,int *jH2,float ***H1,int n)
{
  int i,j,k,l,i1,i2,nH2,it;

  nH2=0;
  for (i=1,i1=1;i<=n;i++) for (j=i;j<=n;j++,i1++)
    {
    for (k=1,i2=1;k<=n;k++) for (l=k;l<=n;l++,i2++)
      {
      if ((i==k)&&(j==l)&&(i==j)) {nH2++;iH2[nH2]=i1;jH2[nH2]=i2;for(it=1;it<2*nt+nt2*dnt2;it++) H2[it][nH2]=2*H1[it][i][i]-anharmonicity;}    /* <2i|V|2i> */

      if ((i==k)&&(j==l)&&(i!=j)) {nH2++;iH2[nH2]=i1;jH2[nH2]=i2;for(it=1;it<2*nt+nt2*dnt2;it++) H2[it][nH2]=H1[it][i][i]+H1[it][j][j];}           /* <1i1j|V|1i1j> */

      if ((i==j)&&(j==l)&&(i!=k)) {nH2++;iH2[nH2]=i1;jH2[nH2]=i2;for(it=1;it<2*nt+nt2*dnt2;it++) H2[it][nH2]=sqrt(2)*H1[it][i][k];}            /* <2i|V|1i1j>  */
      if ((i==j)&&(i==k)&&(j!=l)) {nH2++;iH2[nH2]=i1;jH2[nH2]=i2;for(it=1;it<2*nt+nt2*dnt2;it++) H2[it][nH2]=sqrt(2)*H1[it][i][l];}
      if ((k==l)&&(i==k)&&(j!=l)) {nH2++;iH2[nH2]=i1;jH2[nH2]=i2;for(it=1;it<2*nt+nt2*dnt2;it++) H2[it][nH2]=sqrt(2)*H1[it][j][l];}
      if ((k==l)&&(j==l)&&(i!=k)) {nH2++;iH2[nH2]=i1;jH2[nH2]=i2;for(it=1;it<2*nt+nt2*dnt2;it++) H2[it][nH2]=sqrt(2)*H1[it][i][k];}

      if ((i==k)&&(j!=l)&&(i!=j)&&(k!=l)) {nH2++;iH2[nH2]=i1;jH2[nH2]=i2;for(it=1;it<2*nt+nt2*dnt2;it++) H2[it][nH2]=H1[it][j][l];}             /* <ij|V|kl>  */
      if ((j==l)&&(i!=k)&&(i!=j)&&(k!=l)) {nH2++;iH2[nH2]=i1;jH2[nH2]=i2;for(it=1;it<2*nt+nt2*dnt2;it++) H2[it][nH2]=H1[it][i][k];}
      if ((i==l)&&(j!=k)&&(i!=j)&&(k!=l)) {nH2++;iH2[nH2]=i1;jH2[nH2]=i2;for(it=1;it<2*nt+nt2*dnt2;it++) H2[it][nH2]=H1[it][j][k];}
      if ((j==k)&&(i!=l)&&(i!=j)&&(k!=l)) {nH2++;iH2[nH2]=i1;jH2[nH2]=i2;for(it=1;it<2*nt+nt2*dnt2;it++) H2[it][nH2]=H1[it][i][l];}
      }
    }
  //printf("# of matrix elements in H2:%d\n",nH2);
  //for(i=1;i<=nH2;i++) printf("%4d %4d %6.2f\n",iH2[i],jH2[i],H2[i]);
  return(nH2);
}
/***********************************************************************************/

void fft(complex *P, int n)
/*prepares data for Fouriertransform; i.e. does apodisation, takes care of 1/2 factor for first data point, 
  and sorts the output data such the zero frequency is in the middle*/
{
  int i,j,ir;
  float tmp;
  complex *S;
//float *a;
  
  S=complexvector(0,2*n);
  P[0].re/=2;
  P[0].im/=2;
  for(i=0;i<n;i++)
    {
    tmp=exp(-3.*i*i/n/n);
    P[i].re*=tmp;P[i+n].re=0;
    P[i].im*=tmp;P[i+n].im=0;
    }
  DFT(S,P,2*n);
  for(i=0;i<2*n;i++)
    {
    ir=i+n;
    if(ir>=2*n) ir-=2*n;
    P[i].re=S[ir].re/sqrt(n);
    P[i].im=S[ir].im/sqrt(n);
    }
  P[2*n].re=P[0].re;P[2*n].im=P[0].im;   //symmetrize spectrum
  free_complexvector(S,0,2*n);
  /*
  a=vector(1,4*n);
  for (i=0;i<n;i++)
    {
    a[2*i+1]=P[i].re*exp(-3.*i*i/n/n);a[2*i+1+2*n]=0.;   //second part is zero padding
    a[2*i+2]=P[i].im*exp(-3.*i*i/n/n);a[2*i+2+2*n]=0.;
    }
  
  four1(a,2*n,+1);
  for(i=0;i<2*n;i++)
    {
    ir=i+n;
    if(ir>=2*n) ir-=2*n;
    P[i].re=a[2*ir+1]/sqrt(n);
    P[i].im=a[2*ir+2]/sqrt(n);
    }
  P[2*n].re=P[0].re;P[2*n].im=P[0].im;   //symmetrize spectrum
free_vector(a,1,4*n);
  */
}
/*******************************************************************************/
void DFT(complex *b,complex *a, int n)
// discrete Fourier transform so that any n can be used; not done in a very effcient manner, but will not be the time-limiting
{
  int i,j;
  static int firstmark=0;
  static complex **exp_ikn;

  if(firstmark==0)
    {
    firstmark=1;
    exp_ikn=complexmatrix(0,n-1,0,n-1);
    for(i=0;i<n;i++) for(j=0;j<n;j++) 
      {
      exp_ikn[i][j].re=cos(i*j*2*PI/n);
      exp_ikn[i][j].im=sin(i*j*2*PI/n);
      }
    }

  for(i=0;i<n;i++) for(j=0,b[i].re=0,b[i].im=0;j<n;j++) 
    {
    b[i].re+=exp_ikn[i][j].re*a[j].re-exp_ikn[i][j].im*a[j].im;
    b[i].im+=exp_ikn[i][j].re*a[j].im+exp_ikn[i][j].im*a[j].re;
    }

}

/*******************************************************************************/
void fft2D(complex **R, int n)
{
  int it1,it3,i;
  complex *tmp;

 tmp=complexvector(0,2*n);    //2*n to keep space for zero padding

 for(it3=0;it3<n;it3++) fft(R[it3],n);

 for(it1=0;it1<=2*n;it1++)
  {
  for(it3=0;it3<n;it3++) {tmp[it3].re=R[it3][it1].re; tmp[it3].im=R[it3][it1].im;}
  fft(tmp,n);
  for(it3=0;it3<=2*n;it3++) {R[it3][it1].re=tmp[it3].re; R[it3][it1].im=tmp[it3].im;}
  }

free_complexvector(tmp,0,2*n);
}

/*******************************************************************************/

void write_spectra(complex *S,complex ***Rr,complex ***Rnr)
{
  int i1,i2,i3,j;
  char outfile[50];
  FILE *out;
    
    sprintf(outfile,"spec_lin_%d.dat",mix);
    out=fopen(outfile,"wt");
    for (i1=0;i1<=2*nt;i1++)
      fprintf(out,"%6.1f %8.4f \n",-woff+w0+(i1-nt)*2*PI/nt/2/dt/cw,S[i1].re);
    fclose (out);

    sprintf(outfile,"spec_2D_%d.dat",mix);
    out=fopen(outfile,"wt");
    fprintf(out,"%5.1f\n",dt*dnt2);
    for(i1=0;i1<=2*nt;i1++) for(i3=0;i3<=2*nt;i3++) 
      {
      fprintf(out,"%6.1f %6.1f ",-woff+w0+(i1-nt)*2*PI/nt/2/dt/cw,-woff+w0+(i3-nt)*2*PI/nt/2/dt/cw);
      for(i2=0;i2<nt2;i2++) fprintf(out,"%8.5f ",Rr[i2][i3][2*nt-i1].re+Rnr[i2][i3][i1].re);  
      fprintf(out,"\n");
      }
    fclose(out);
   
}





/***********************************************************************************/
/*                       Numerical Recipis Routines                                */
/***********************************************************************************/
#define EPS 1.0e-16
#define FPMIN 1.0e-30
#define MAXIT 10000
#define XMIN 2.0
static int imaxarg1,imaxarg2;
#define SIGN(a,b) ((b) >= 0.0 ? fabs(a) : -fabs(a))
#define IMAX(a,b) (imaxarg1=(a),imaxarg2=(b),(imaxarg1) > (imaxarg2) ?\
        (imaxarg1) : (imaxarg2))

void bessjy(double x, double xnu, double *rj, double *ry, double *rjp, double *ryp)
{
    void beschb(double x, double *gam1, double *gam2, double *gampl,
		double *gammi);
    int i,isign,l,nl;
    double a,b,br,bi,c,cr,ci,d,del,del1,den,di,dlr,dli,dr,e,f,fact,fact2,
	fact3,ff,gam,gam1,gam2,gammi,gampl,h,p,pimu,pimu2,q,r,rjl,
	rjl1,rjmu,rjp1,rjpl,rjtemp,ry1,rymu,rymup,rytemp,sum,sum1,
	temp,w,x2,xi,xi2,xmu,xmu2;

    if (x <= 0.0 || xnu < 0.0) nrerror("bad arguments in bessjy");
    nl=(x < XMIN ? (int)(xnu+0.5) : IMAX(0,(int)(xnu-x+1.5)));
    xmu=xnu-nl;
    xmu2=xmu*xmu;
    xi=1.0/x;
    xi2=2.0*xi;
    w=xi2/PI;
    isign=1;
    h=xnu*xi;
    if (h < FPMIN) h=FPMIN;
    b=xi2*xnu;
    d=0.0;
    c=h;
    for (i=1;i<=MAXIT;i++) {
	b += xi2;
	d=b-d;
	if (fabs(d) < FPMIN) d=FPMIN;
	c=b-1.0/c;
	if (fabs(c) < FPMIN) c=FPMIN;
	d=1.0/d;
	del=c*d;
	h=del*h;
	if (d < 0.0) isign = -isign;
	if (fabs(del-1.0) < EPS) break;
    }
    if (i > MAXIT) nrerror("x too large in bessjy; try asymptotic expansion");
    rjl=isign*FPMIN;
    rjpl=h*rjl;
    rjl1=rjl;
    rjp1=rjpl;
    fact=xnu*xi;
    for (l=nl;l>=1;l--) {
	rjtemp=fact*rjl+rjpl;
	fact -= xi;
	rjpl=fact*rjtemp-rjl;
	rjl=rjtemp;
    }
    if (rjl == 0.0) rjl=EPS;
    f=rjpl/rjl;
    if (x < XMIN) {
	x2=0.5*x;
	pimu=PI*xmu;
	fact = (fabs(pimu) < EPS ? 1.0 : pimu/sin(pimu));
	d = -log(x2);
	e=xmu*d;
	fact2 = (fabs(e) < EPS ? 1.0 : sinh(e)/e);
	beschb(xmu,&gam1,&gam2,&gampl,&gammi);
	ff=2.0/PI*fact*(gam1*cosh(e)+gam2*fact2*d);
	e=exp(e);
	p=e/(gampl*PI);
	q=1.0/(e*PI*gammi);
	pimu2=0.5*pimu;
	fact3 = (fabs(pimu2) < EPS ? 1.0 : sin(pimu2)/pimu2);
	r=PI*pimu2*fact3*fact3;
	c=1.0;
	d = -x2*x2;
	sum=ff+r*q;
	sum1=p;
	for (i=1;i<=MAXIT;i++) {
	    ff=(i*ff+p+q)/(i*i-xmu2);
	    c *= (d/i);
	    p /= (i-xmu);
	    q /= (i+xmu);
	    del=c*(ff+r*q);
	    sum += del;
	    del1=c*p-i*del;
	    sum1 += del1;
	    if (fabs(del) < (1.0+fabs(sum))*EPS) break;
	}
	if (i > MAXIT) nrerror("bessy series failed to converge");
	rymu = -sum;
	ry1 = -sum1*xi2;
	rymup=xmu*xi*rymu-ry1;
	rjmu=w/(rymup-f*rymu);
    } else {
	a=0.25-xmu2;
	p = -0.5*xi;
	q=1.0;
	br=2.0*x;
	bi=2.0;
	fact=a*xi/(p*p+q*q);
	cr=br+q*fact;
	ci=bi+p*fact;
	den=br*br+bi*bi;
	dr=br/den;
	di = -bi/den;
	dlr=cr*dr-ci*di;
	dli=cr*di+ci*dr;
	temp=p*dlr-q*dli;
	q=p*dli+q*dlr;
	p=temp;
	for (i=2;i<=MAXIT;i++) {
	    a += 2*(i-1);
	    bi += 2.0;
	    dr=a*dr+br;
	    di=a*di+bi;
	    if (fabs(dr)+fabs(di) < FPMIN) dr=FPMIN;
	    fact=a/(cr*cr+ci*ci);
	    cr=br+cr*fact;
	    ci=bi-ci*fact;
	    if (fabs(cr)+fabs(ci) < FPMIN) cr=FPMIN;
	    den=dr*dr+di*di;
	    dr /= den;
	    di /= -den;
	    dlr=cr*dr-ci*di;
	    dli=cr*di+ci*dr;
	    temp=p*dlr-q*dli;
	    q=p*dli+q*dlr;
	    p=temp;
	    if (fabs(dlr-1.0)+fabs(dli) < EPS) break;
	}
	if (i > MAXIT) nrerror("cf2 failed in bessjy");
	gam=(p-f)/q;
	rjmu=sqrt(w/((p-f)*gam+q));
	rjmu=SIGN(rjmu,rjl);
	rymu=rjmu*gam;
	rymup=rymu*(p+q/gam);
	ry1=xmu*xi*rymu-rymup;
    }
    fact=rjmu/rjl;
    *rj=rjl1*fact;
    *rjp=rjp1*fact;
    for (i=1;i<=nl;i++) {
	rytemp=(xmu+i)*xi2*ry1-rymu;
	rymu=ry1;
	ry1=rytemp;
    }
    *ry=rymu;
    *ryp=xnu*xi*rymu-ry1;
}
#undef EPS
#undef FPMIN
#undef MAXIT
#undef XMIN

#define NUSE1 7
#define NUSE2 8

void beschb(double x, double *gam1, double *gam2, double *gampl, double *gammi)
{
    double chebev(double a, double b, double c[], int m, double x);
    double xx;
    static double c1[] = {
	-1.142022680371172e0,6.516511267076e-3,
	3.08709017308e-4,-3.470626964e-6,6.943764e-9,
	3.6780e-11,-1.36e-13};
    static double c2[] = {
	1.843740587300906e0,-0.076852840844786e0,
	1.271927136655e-3,-4.971736704e-6,-3.3126120e-8,
	2.42310e-10,-1.70e-13,-1.0e-15};

    xx=8.0*x*x-1.0;
    *gam1=chebev(-1.0,1.0,c1,NUSE1,xx);
    *gam2=chebev(-1.0,1.0,c2,NUSE2,xx);
    *gampl= *gam2-x*(*gam1);
    *gammi= *gam2+x*(*gam1);
}
#undef NUSE1
#undef NUSE2

double chebev(double a, double b, double c[], int m, double x)
{
    void nrerror(char error_text[]);
    double d=0.0,dd=0.0,sv,y,y2;
    int j;

    if ((x-a)*(x-b) > 0.0) nrerror("x not in range in routine chebev");
    y2=2.0*(y=(2.0*x-a-b)/(b-a));
    for (j=m-1;j>=1;j--) {
	sv=d;
	d=y2*d-dd+c[j];
	dd=sv;
    }
    return y*d-dd+0.5*c[0];
}

/*************************************************/



#define SWAP(a,b) tempr=(a);(a)=(b);(b)=tempr

void four1(float data[], unsigned long nn, int isign)
{
        unsigned long n,mmax,m,j,istep,i;
        double wtemp,wr,wpr,wpi,wi,theta;
        float tempr,tempi;

        n=nn << 1;
        j=1;
        for (i=1;i<n;i+=2) {
                if (j > i) {
                        SWAP(data[j],data[i]);
                        SWAP(data[j+1],data[i+1]);
                }
                m=n >> 1;
                while (m >= 2 && j > m) {
                        j -= m;
                        m >>= 1;
                }
                j += m;
        }
        mmax=2;
        while (n > mmax) {
                istep=mmax << 1;
                theta=isign*(6.28318530717959/mmax);
                wtemp=sin(0.5*theta);
                wpr = -2.0*wtemp*wtemp;
                wpi=sin(theta);
                wr=1.0;
                wi=0.0;
                for (m=1;m<mmax;m+=2) {
                        for (i=m;i<=n;i+=istep) {
                                j=i+mmax;
                                tempr=wr*data[j]-wi*data[j+1];
                                tempi=wr*data[j+1]+wi*data[j];
                                data[j]=data[i]-tempr;
                                data[j+1]=data[i+1]-tempi;
                                data[i] += tempr;
                                data[i+1] += tempi;
                        }
                        wr=(wtemp=wr)*wpr-wi*wpi+wr;
                        wi=wi*wpr+wtemp*wpi+wi;
                }
                mmax=istep;
        }
}
#undef SWAP
/* (C) Copr. 1986-92 Numerical Recipes Software ,2:. */

/*********************************************************************/




#define NRANSI
static float sqrarg;
#define SQR(a) ((sqrarg=(a)) == 0.0 ? 0.0 : sqrarg*sqrarg)
#define SIGN(a,b) ((b) >= 0.0 ? fabs(a) : -fabs(a))

void tqli(float d[], float e[], int n, float **z)
{
        float pythag(float a, float b);
        int m,l,iter,i,k;
        float s,r,p,g,f,dd,c,b;

        for (i=2;i<=n;i++) e[i-1]=e[i];
        e[n]=0.0;
        for (l=1;l<=n;l++) {
                iter=0;
                do {
		  for (m=l;m<=n-1;m++) {
                                dd=fabs(d[m])+fabs(d[m+1]);
                                if ((float)(fabs(e[m])+dd) == dd) break;
		  }
		  if (m != l) {
                                if (iter++ == 30) nrerror("Too many iterations in tqli");
                                g=(d[l+1]-d[l])/(2.0*e[l]);
                                r=pythag(g,1.0);
                                g=d[m]-d[l]+e[l]/(g+SIGN(r,g));
                                s=c=1.0;
                                p=0.0;
                                for (i=m-1;i>=l;i--) {
                                        f=s*e[i];
                                        b=c*e[i];
                                        e[i+1]=(r=pythag(f,g));
                                        if (r == 0.0) {
                                                d[i+1] -= p;
                                                e[m]=0.0;
                                                break;
                                        }
                                        s=f/r;
                                        c=g/r;
                                        g=d[i+1]-p;
                                        r=(d[i]-g)*s+2.0*c*b;
                                        d[i+1]=g+(p=s*r);
                                        g=c*r-b;
                                        for (k=1;k<=n;k++) {
                                                f=z[k][i+1];
                                                z[k][i+1]=s*z[k][i]+c*f;
                                                z[k][i]=c*z[k][i]-s*f;
                                        }
                                }
                                if (r == 0.0 && i >= l) continue;
                                d[l] -= p;
                                e[l]=g;
                                e[m]=0.0;
		  }
                } while (m != l);
        }
}

/********************************************************************/

float pythag(float a, float b)
{
        float absa,absb;
        absa=fabs(a);
        absb=fabs(b);
        if (absa > absb) return absa*sqrt(1.0+SQR(absb/absa));
        else return (absb == 0.0 ? 0.0 : absb*sqrt(1.0+SQR(absa/absb)));
}

/*********************************************************************/
void tred2(float **a, int n, float d[], float e[])
{
        int l,k,j,i;
        float scale,hh,h,g,f;

        for (i=n;i>=2;i--) {
                l=i-1;
                h=scale=0.0;
                if (l > 1) {
                        for (k=1;k<=l;k++)
                                scale += fabs(a[i][k]);
                        if (scale == 0.0)
                                e[i]=a[i][l];
                        else {
			  for (k=1;k<=l;k++) {
                                        a[i][k] /= scale;
                                        h += a[i][k]*a[i][k];
			  }
                                f=a[i][l];
                                g=(f >= 0.0 ? -sqrt(h) : sqrt(h));
                                e[i]=scale*g;
                                h -= f*g;
                                a[i][l]=f-g;
                                f=0.0;
                                for (j=1;j<=l;j++) {
                                        a[j][i]=a[i][j]/h;
                                        g=0.0;
                                        for (k=1;k<=j;k++)
                                                g += a[j][k]*a[i][k];
                                        for (k=j+1;k<=l;k++)
                                                g += a[k][j]*a[i][k];
                                        e[j]=g/h;
                                        f += e[j]*a[i][j];
                                }
                                hh=f/(h+h);
                                for (j=1;j<=l;j++) {
                                        f=a[i][j];
                                        e[j]=g=e[j]-hh*f;
                                        for (k=1;k<=j;k++)
                                                a[j][k] -= (f*e[k]+g*a[i][k]);
                                }
                        }
                } else
                        e[i]=a[i][l];
                d[i]=h;
        }
        d[1]=0.0;
        e[1]=0.0;
        /* Contents of this loop can be omitted if eigenvectors not
                        wanted except for statement d[i]=a[i][i]; */
        for (i=1;i<=n;i++) {
                l=i-1;
                if (d[i]) {
		  for (j=1;j<=l;j++) {
                                g=0.0;
                                for (k=1;k<=l;k++)
                                        g += a[i][k]*a[k][j];
                                for (k=1;k<=l;k++)
                                        a[k][j] -= g*a[k][i];
		  }
                }
                d[i]=a[i][i];
                a[i][i]=1.0;
                for (j=1;j<=l;j++) a[j][i]=a[i][j]=0.0;
        }
}







/* (C) Copr. 1986-92 Numerical Recipes Software ,2:. */

/***********************************************************************************/

void eigsrt(float d[], float **v, int n)
{
	int k,j,i;
	float p;

	for (i=1;i<n;i++) {
		p=d[k=i];
		for (j=i+1;j<=n;j++)
			if (d[j] >= p) p=d[k=j];
		if (k != i) {
			d[k]=d[i];
			d[i]=p;
			for (j=1;j<=n;j++) {
				p=v[j][i];
				v[j][i]=v[j][k];
				v[j][k]=p;
			}
		}
	}
}
/* (C) Copr. 1986-92 Numerical Recipes Software ,2:. */

/*************************************************************************************/
float gasdev(long *idum)
{

static int iset=0;
static float gset;
float fac,rsq,v1,v2;

	          if  (iset == 0) {
		                    do {
				    v1=2.0*ran1(idum)-1.0;
	                            v2=2.0*ran1(idum)-1.0;
	                            rsq=v1*v1+v2*v2;
				        } while (rsq >= 1.0 || rsq == 0.0);
                                    fac=sqrt(-2.0*log(rsq)/rsq);
 		                    gset=v1*fac;
		                    iset=1;
			            return v2*fac;
		    } else {
			                 iset=0;
	       	                         return gset;
	            }
}

/************************************************************************************/
/******************************************/
#define IA 16807
#define IM 2147483647
#define AM (1.0/IM)
#define IQ 127773
#define IR 2836
#define MASK 123459876

float ran0(long *idum)
{
        long k;
        float ans;

        *idum ^= MASK;
        k=(*idum)/IQ;
        *idum=IA*(*idum-k*IQ)-IR*k;
        if (*idum < 0) *idum += IM;
        ans=AM*(*idum);
        *idum ^= MASK;
        return ans;
}
#undef IA
#undef IM
#undef AM
#undef IQ
#undef IR
#undef MASK
/****************************************/

#define IA 16807
#define IM 2147483647
#define AM (1.0/IM)
#define IQ 127773
#define IR 2836
#define NTAB 32
#define NDIV (1+(IM-1)/NTAB)
#define EPS 1.2e-7
#define RNMX (1.0-EPS)

float ran1(long *idum)
{
          int j;
	  long k;
	  static long iy=0;
	  static long iv[NTAB];
	  float temp;

	  if (*idum <= 0 || !iy) {
		  if (-(*idum) < 1) *idum=1;
		  else *idum = -(*idum);
                  for (j=NTAB+7;j>=0;j--) {
		        k=(*idum)/IQ;
                        *idum=IA*(*idum-k*IQ)-IR*k;
                        if (*idum < 0) *idum += IM;
                        if (j < NTAB) iv[j] = *idum;
					   }
	          iy=iv[0];
		   }
	          k=(*idum)/IQ;
	          *idum=IA*(*idum-k*IQ)-IR*k;
	          if (*idum < 0) *idum += IM;
	          j=iy/NDIV;
	          iy=iv[j];
	          iv[j] = *idum;
	          if ((temp=AM*iy) > RNMX) return RNMX;
	          else return temp;
	}
#undef IA
#undef IM
#undef AM
#undef IQ
#undef IR
#undef NTAB
#undef NDIV
#undef EPS
#undef RNMX

/*************************************************************************************/


#define NR_END 1

void nrerror(char error_text[])
/* Numerical Recipes standard error handler */
{
        printf("Numerical Recipes run-time error...\n");
        printf("%s\n",error_text);
        printf("...now exiting to system...\n");
        exit(1);
      }

float *vector(long nl, long nh)
/* allocate a float vector with subscript range v[nl..nh] */
{
        float *v;

        v=(float *)malloc((size_t) ((nh-nl+1+NR_END)*sizeof(float)));
        if (!v) nrerror("allocation failure in vector()");
        return (v-nl+NR_END);
      }

float **matrix(long nrl, long nrh, long ncl, long nch)
/* allocate a float matrix with subscript range m[nrl..nrh][ncl..nch] */
{
        long i, nrow=nrh-nrl+1,ncol=nch-ncl+1;
        float **m;

        /* allocate pointers to rows */
        m=(float **) malloc((size_t)((nrow+NR_END)*sizeof(float*)));
        if (!m) nrerror("allocation failure 1 in matrix()");
        m += NR_END;
        m -= nrl;

        /* allocate rows and set pointers to them */
        m[nrl]=(float *) malloc((size_t)((nrow*ncol+NR_END)*sizeof(float)));
        if (!m[nrl]) nrerror("allocation failure 2 in matrix()");
        m[nrl] += NR_END;
        m[nrl] -= ncl;

        for(i=nrl+1;i<=nrh;i++) m[i]=m[i-1]+ncol;

        /* return pointer to array of pointers to rows */
        return (m);
      }

float ***f3tensor(long nrl, long nrh, long ncl, long nch, long ndl, long ndh)
     /* allocate a float 3tensor with range t[nrl..nrh][ncl..nch][ndl..ndh] */
{
        long i,j,nrow=nrh-nrl+1,ncol=nch-ncl+1,ndep=ndh-ndl+1;
        float ***t;

        /* allocate pointers to pointers to rows */
        t=(float ***) malloc((size_t)((nrow+NR_END)*sizeof(float**)));
        if (!t) nrerror("allocation failure 1 in f3tensor()");
        t += NR_END;
        t -= nrl;

        /* allocate pointers to rows and set pointers to them */
        t[nrl]=(float **) malloc((size_t)((nrow*ncol+NR_END)*sizeof(float*)));
        if (!t[nrl]) nrerror("allocation failure 2 in f3tensor()");
        t[nrl] += NR_END;
        t[nrl] -= ncl;

        /* allocate rows and set pointers to them */
        t[nrl][ncl]=(float *) malloc((size_t)((nrow*ncol*ndep+NR_END)*sizeof(float)));
        if (!t[nrl][ncl]) nrerror("allocation failure 3 in f3tensor()");
        t[nrl][ncl] += NR_END;
        t[nrl][ncl] -= ndl;

        for(j=ncl+1;j<=nch;j++) t[nrl][j]=t[nrl][j-1]+ndep;
        for(i=nrl+1;i<=nrh;i++) {
                t[i]=t[i-1]+ncol;
                t[i][ncl]=t[i-1][ncl]+ncol*ndep;
                for(j=ncl+1;j<=nch;j++) t[i][j]=t[i][j-1]+ndep;
        }

        /* return pointer to array of pointers to rows */
        return t;
}


void free_vector(float *v, long nl, long nh)
/* free a float vector allocated with vector() */
{
        nh++;
        free((char*) (v+nl-NR_END));
      }


void free_matrix(float **m, long nrl, long nrh, long ncl, long nch)
/* free a float matrix allocated by matrix() */
{
        nch++;nrh++;
        free((char*) (m[nrl]+ncl-NR_END));
        free((char*) (m+nrl-NR_END));
      }

void free_complexmatrix(complex **m, long nrl, long nrh, long ncl, long nch)
/* free a float matrix allocated by matrix() */
{
        nch++;nrh++;
        free((char*) (m[nrl]+ncl-NR_END));
        free((char*) (m+nrl-NR_END));
      }


int **imatrix(long nrl, long nrh, long ncl, long nch)
/* allocate a int matrix with subscript range m[nrl..nrh][ncl..nch] */
{
        long i, nrow=nrh-nrl+1,ncol=nch-ncl+1;
        int **m;

        /* allocate pointers to rows */
        m=(int **) malloc((size_t)((nrow+NR_END)*sizeof(int*)));
        if (!m) nrerror("allocation failure 1 in matrix()");
        m += NR_END;
        m -= nrl;


        /* allocate rows and set pointers to them */
        m[nrl]=(int *) malloc((size_t)((nrow*ncol+NR_END)*sizeof(int)));
        if (!m[nrl]) nrerror("allocation failure 2 in matrix()");
        m[nrl] += NR_END;
        m[nrl] -= ncl;

        for(i=nrl+1;i<=nrh;i++) m[i]=m[i-1]+ncol;

        /* return pointer to array of pointers to rows */
        return m;
      }

int *ivector(long nl, long nh)
     /* allocate an int vector with subscript range v[nl..nh] */
{
        int *v;

        v=(int *)malloc((size_t) ((nh-nl+1+NR_END)*sizeof(int)));
        if (!v) nrerror("allocation failure in ivector()");
        return v-nl+NR_END;
}

double *dvector(long nl, long nh)
     /* allocate an int vector with subscript range v[nl..nh] */
{
        double *v;

        v=(double *)malloc((size_t) ((nh-nl+1+NR_END)*sizeof(double)));
        if (!v) nrerror("allocation failure in ivector()");
        return v-nl+NR_END;
}


void free_ivector(int *v, long nl, long nh)
     /* free an int vector allocated with ivector() */
{
        free((char*)(v+nl-NR_END));
}


void free_imatrix(int **m,long nrl,long nrh,long ncl,long nch)
/* free an int matrix allocated by imatrix() */
{
        nch++;nrh++;
        free((char*) (m[nrl]+ncl-NR_END));
        free((char*) (m+nrl-NR_END));
      }


void free_f3tensor(float ***t, long nrl, long nrh, long ncl, long nch,
        long ndl, long ndh)
     /* free a float f3tensor allocated by f3tensor() */
{
        free((char*) (t[nrl][ncl]+ndl-NR_END));
        free((char*) (t[nrl]+ncl-NR_END));
        free((char*) (t+nrl-NR_END));
}

complex **complexmatrix(long nrl, long nrh, long ncl, long nch)
/* allocate a complex matrix with subscript range m[nrl..nrh][ncl..nch] */
{
        long i, nrow=nrh-nrl+1,ncol=nch-ncl+1;
        complex **m;

        /* allocate pointers to rows */
        m=(complex **) malloc((size_t)((nrow+NR_END)*sizeof(complex*)));
        if (!m) nrerror("allocation failure 1 in matrix()");
        m += NR_END;
        m -= nrl;

        /* allocate rows and set pointers to them */
        m[nrl]=(complex *) malloc((size_t)((nrow*ncol+NR_END)*sizeof(complex)));
        if (!m[nrl]) nrerror("allocation failure 2 in matrix()");
        m[nrl] += NR_END;
        m[nrl] -= ncl;

        for(i=nrl+1;i<=nrh;i++) m[i]=m[i-1]+ncol;

        /* return pointer to array of pointers to rows */
        return (m);
      }


complex *complexvector(long nl, long nh)
/* allocate a double vector with subscript range v[nl..nh] */
{
    complex *v;

    v=(complex *)malloc((size_t) ((nh-nl+1+NR_END)*sizeof(complex)));
    if (!v) nrerror("allocation failure in dvector()");
    return v-nl+NR_END;
}


void free_complexvector(complex *v, long nl, long nh)
{
        free((char*)(v+nl-NR_END));
}




/*
complex ****complex4tensor(long nrl, long nrh, long ncl, long nch, long ndl, long ndh, long nel, long neh)
{
  long i,j,k,nrow=nrh-nrl+1,ncol=nch-ncl+1,ndep=ndh-ndl+1,ndep2=neh-nel+1;
  complex ****t;

// allocate pointers to pointers to rows 
  t=(complex ****) malloc((size_t)((nrow+NR_END)*sizeof(complex***)));
  if (!t) nrerror("allocation failure 1 in f4tensor()");
  t += NR_END;
  t -= nrl;

// allocate pointers to rows and set pointers to them 
  t[nrl]=(complex ***) malloc((size_t)((nrow*ncol+NR_END)*sizeof(complex**)));
  if (!t[nrl]) nrerror("allocation failure 2 in f4tensor()");
  t[nrl] += NR_END;
  t[nrl] -= ncl;

// allocate rows and set pointers to them 
  t[nrl][ncl]=(complex **) malloc((size_t)((nrow*ncol*ndep+NR_END)*sizeof(complex*)));
  if (!t[nrl][ncl]) nrerror("allocation failure 3 in f4tensor()");
  t[nrl][ncl] += NR_END;
  t[nrl][ncl] -= ndl;

  t[nrl][ncl][ndl]=(complex *) malloc((size_t)((nrow*ncol*ndep*ndep2+NR_END)*sizeof(complex)));
  if (!t[nrl][ncl][ndl]) nrerror("allocation failure 3 in f4tensor()");
  t[nrl][ncl][ndl] += NR_END;
  t[nrl][ncl][ndl] -= nel;




 for(j=ndl+1;j<=ndh;j++) t[nrl][ncl][j]=t[nrl][ncl][j-1]+ndep2;
 for(i=ncl+1;i<=nch;i++) {
    t[nrl][i]=t[nrl][i-1]+ndep;
    t[nrl][i][ndl]=t[nrl][i-1][ndl]+ndep*ndep2;
    for(j=ndl+1;j<=ndh;j++) t[nrl][i][j]=t[nrl][i][j-1]+ndep2;
  }

  for(i=nrl+1;i<=nrh;i++) {
    t[i]=t[i-1]+ncol;
    t[i][ncl]=t[i-1][ncl]+ncol*ndep;
    t[i][ncl][ndl]=t[i-1][ncl][ndl]+ncol*ndep*ndep2;
    for(j=ndl+1;j<=ndh;j++) t[i][ncl][j]=t[i][ncl][j-1]+ndep2;
    for(j=ncl+1;j<=nch;j++) 
      {
      t[i][j]=t[i][j-1]+ndep;
      t[i][j][ndl]=t[i][j-1][ndl]+ndep*ndep2;
      for(k=ndl+1;k<=ndh;k++) t[i][j][k]=t[i][j][k-1]+ndep2;
      }
  }


// return pointer to array of pointers to rows 
  return t;
}
*/

complex ****complex4tensor(long nrl, long nrh, long ncl, long nch, long ndl, long ndh, long nwl, long nwh)
/* allocate a complex 4tensor with range t[nrl..nrh][ncl..nch][ndl..ndh][nwl..nwh] */
// taken from http://numerical.recipes/forum/showthread.php?t=450
{
	long i,j,k, nrow=nrh-nrl+1,ncol=nch-ncl+1,ndep=ndh-ndl+1,nwid=nwh-nwl+1 ;
	complex ****t;

	/* allocate pointers to pointers to pointers to rows */
	t=(complex ****) malloc((size_t)((nrow+NR_END)*sizeof(complex***)));
	if (!t) nrerror("allocation failure 1 in f4tensor()");
	t += NR_END;
	t -= nrl;

	/* allocate pointers to pointers to rows and set pointers to them */
	t[nrl]=(complex ***) malloc((size_t)((nrow*ncol+NR_END)*sizeof(complex**)));
	if (!t[nrl]) nrerror("allocation failure 2 in f4tensor()");
	t[nrl] += NR_END;
	t[nrl] -= ncl;

	/* allocate pointers to rows and set pointers to them */
	t[nrl][ncl]=(complex **) malloc((size_t)((nrow*ncol*ndep+NR_END)*sizeof(complex*)));
	if (!t[nrl][ncl]) nrerror("allocation failure 3 in f4tensor()");
	t[nrl][ncl] += NR_END;
	t[nrl][ncl] -= ndl;

	/* allocate rows and set pointers to them */
	t[nrl][ncl][ndl]=(complex *) malloc((size_t)((nrow*ncol*ndep*nwid+NR_END)*sizeof(complex)));
	if (!t[nrl][ncl][ndl]) nrerror("allocation failure 4 in f4tensor()");
	t[nrl][ncl][ndl] += NR_END;
	t[nrl][ncl][ndl] -= nwl;

    for(i=nrl;i<=nrh;i++)
	{
		if (i > nrl)
		{
		    t[i] = t[i-1] + ncol ;
		    t[i][ncl]=t[i-1][ncl]+ncol*ndep;
		    t[i][ncl][ndl] = t[i-1][ncl][ndl] + ncol*ndep*nwid ;
		}
		for(j=ncl;j<=nch;j++)
		{
			if (j > ncl)
			{
				t[i][j]=t[i][j-1] + ndep ;
				t[i][j][ndl] = t[i][j-1][ndl] + ndep*nwid ;
			}

			for(k=ndl;k<=ndh;k++)
			{
				if (k > ndl) t[i][j][k] = t[i][j][k-1] + nwid ;
			}
		}
	}

	/* return pointer to pointer to array of pointers to rows */
	return t;
}


complex ***complex3tensor(long nrl, long nrh, long ncl, long nch, long ndl, long ndh)
{
    long i,j,nrow=nrh-nrl+1,ncol=nch-ncl+1,ndep=ndh-ndl+1;
    complex ***t;

    /* allocate pointers to pointers to rows */
    t=(complex ***) malloc((size_t)((nrow+NR_END)*sizeof(complex**)));
    if (!t) nrerror("allocation failure 1 in f3tensor()");
    t += NR_END;
    t -= nrl;

    /* allocate pointers to rows and set pointers to them */
    t[nrl]=(complex **) malloc((size_t)((nrow*ncol+NR_END)*sizeof(complex*)));
    if (!t[nrl]) nrerror("allocation failure 2 in f3tensor()");
    t[nrl] += NR_END;
    t[nrl] -= ncl;

    /* allocate rows and set pointers to them */
    t[nrl][ncl]=(complex *) malloc((size_t)((nrow*ncol*ndep+NR_END)*sizeof(complex)));
    if (!t[nrl][ncl]) nrerror("allocation failure 3 in f3tensor()");
    t[nrl][ncl] += NR_END;
    t[nrl][ncl] -= ndl;

    for(j=ncl+1;j<=nch;j++) t[nrl][j]=t[nrl][j-1]+ndep;
    for(i=nrl+1;i<=nrh;i++) {
        t[i]=t[i-1]+ncol;
        t[i][ncl]=t[i-1][ncl]+ncol*ndep;
        for(j=ncl+1;j<=nch;j++) t[i][j]=t[i][j-1]+ndep;
    }

    /* return pointer to array of pointers to rows */
    return t;
}
