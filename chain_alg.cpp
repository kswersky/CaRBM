/*
Code provided by Kevin Swersky, Danny Tarlow, Ilya Sutskever, Ruslan Salakhutdinov, Rich Zemel and Ryan Adams.

Permission is granted for anyone to copy, use, modify, or distribute this
program and accompanying programs and documents for any purpose, provided
this copyright notice is retained and prominently displayed, along with
a note saying that the original programs are available from our 
web page.

The programs and documents are distributed without any warranty, express or
implied.  As the programs were written for research purposes only, they have
not been tested to the degree that would be advisable in any important
application.  All use of these programs is entirely at the user's own risk.

This code implements the methods described in the paper:
Cardinality Restricted Boltzmann Machines. NIPS 2012.
*/

#include <iostream>
 #include <stdio.h>
 #include <cstdlib>

 using namespace std;

 void print_msgs(int D, int Kmax, double *msgs) {
   int K = Kmax+1;

   for (int d = 0; d < D; d++) {
     int dK = d*K;
     for (int k = 0; k < K; k++) {
       cout << msgs[dK+k] << " ";
     }
     cout << endl;
   }
 }


 int sample_categorical(double *unnorm_probs, int K) {
   int cur = -1;
   double psum = 0;
   for (int k = 0; k < K; k++) {
     psum += unnorm_probs[k];
     if (unnorm_probs[k]/psum > (rand() / static_cast<double>(RAND_MAX)))  cur = k;
   }
   return cur;
 }


 extern "C" {
   /* compute marginals and draw a sample for each of the B instances
      in the minibatch. */
   void infer(int D, int Kmin, int Kmax, int B, double *exp_node_pots,
	      double *fmsgs, double *bmsgs, double *buf,
	      double *marginals, bool *samples) {

     // fmsgs and bmsgs don't need to have meaningful values in them, but
     // they need to have allocated space for D*(Kmax+1) doubles
     int K = Kmax+1;

     for (int b = 0; b < B; b++) {
       double *exp_node_pots_b = exp_node_pots + b*D;

       // backward messages
       bmsgs[(D-1)*K+0] = 1; 
       bmsgs[(D-1)*K+1] = exp_node_pots_b[D-1]; 
       for (int k = 2; k < K; k++)  bmsgs[(D-1)*K+k] = 0;
       for (int d = D-1; d > 0; d--) {
	 int dK = d*K;
	 double sum = 0;
	 for (int k = 0; k < K; k++) {
	   bmsgs[dK-K+k] = bmsgs[dK+k] + exp_node_pots_b[d-1] * ((k==0) ? 0 : bmsgs[dK+k-1]);
	   sum += bmsgs[dK-K+k];
	 }
	 for (int k = 0; k < K; k++)  bmsgs[dK-K+k] /= sum;
       }

       // forward messages
       for (int k = 0; k < K; k++)  fmsgs[k] = k >= Kmin ? 1 : 0;
       for (int d = 0; d < D-1; d++) {
	 int dK = d*K;
	 double sum = 0;
	 for (int k = 0; k < K; k++) {
	   fmsgs[dK+K+k] = fmsgs[dK+k] + exp_node_pots_b[d] * ((k==K-1) ? 0 : fmsgs[dK+k+1]);
	   sum += fmsgs[dK+K+k];
	 }
	 for (int k = 0; k < K; k++)  fmsgs[dK+K+k] /= sum;  // normalize
       }

       // forward sample
       double *count_belief = buf;
       for (int k = 0; k < K; k++)  count_belief[k] = fmsgs[k]*bmsgs[k];

       int ct = sample_categorical(count_belief, K);
       for (int d = 0; d < D-1; d++) {
	 if (ct == 0)  samples[b*D+d] = 0;
	 else {
	   double p0 = bmsgs[d*K+K+ct];
	   double p1 = exp_node_pots_b[d] * bmsgs[d*K+K+ct-1];

	   samples[b*D+d] = (p1/(p0+p1)) > (rand() / static_cast<double>(RAND_MAX));
	   ct -= samples[b*D+d];
	 }
       }
       if (ct == 0 || ct == 1)  samples[b*D+D-1] = ct;
       //else  assert(false);

       // compute marginals
       double b0;
       double b1;
       for (int d = 0; d < D-1; d++) {
	 int dK = d*K;

	 b0 = 0; b1 = 0;

	 for (int k = 0; k < K; k++)    b0 += bmsgs[dK+K+k] * fmsgs[dK+k];
	 for (int k = 0; k < K-1; k++)  b1 += bmsgs[dK+K+k] * fmsgs[dK+k+1];
	 b1 *= exp_node_pots_b[d];

	marginals[b*D+d] = b1/(b0+b1);
      }
      b0 = fmsgs[(D-1)*K] * bmsgs[(D-1)*K];
      b1 = fmsgs[(D-1)*K+1] * bmsgs[(D-1)*K+1];
      marginals[b*D+D-1] = b1/(b0+b1);
    }
  }
}


int main(int argc, char **argv) {

  int D = atoi(argv[1]);      // num vars
  int Kmin = atoi(argv[2]);   // force between Kmin and Kmax on
  int Kmax = atoi(argv[3]);   // force between Kmin and Kmax on
  int seed = 0; //atoi(argv[4]);
  int B    = 10;             // size of minibatch

  srand(seed);

  int msgs_size = D*(Kmax+1);
  double *fmsgs = new double[msgs_size];
  double *bmsgs = new double[msgs_size];
  double *buf = new double[D+1];
  double *exp_node_pots = new double[D*B];
  double *marginals = new double[D*B];
  bool *samples = new bool[D*B];

  for (int i = 0; i < D*B; i++) {
    exp_node_pots[i] = 1; //(.0001 + (i % D)) / D;
    //exp_node_pots[i] = rand() / static_cast<double>(RAND_MAX);
  }

  infer(D, Kmin, Kmax, B, exp_node_pots, fmsgs, bmsgs, buf, marginals, samples);

  if (false) {
    cout << "Forward messages" << endl;
    print_msgs(D,Kmax,fmsgs);
    
    cout << "Backward messages" << endl;
    print_msgs(D,Kmax,bmsgs);
  }

  cout << "Samples" << endl;
  double ct_ct[D+1];    
  for (int d = 0; d < D+1; d++) ct_ct[d] = 0;
  for (int b = 0; b < B; b++) {
    for (int d = 0; d < D; d++)  cout << samples[b*D+d] << " ";  cout << endl;
    int ct = 0;
    for (int d = 0; d < D; d++) ct += samples[b*D+d];
    ct_ct[ct]++;
  }
  cout << "Count sample marginals" << endl;
  for (int d = 0; d < D+1; d++)  cout << ct_ct[d] << " ";  cout << endl;

  cout << "Marginals" << endl;
  for (int b = 0; b < B; b++) {
    for (int d = 0; d < D; d++)  cout << marginals[b*D+d] << " ";  cout << endl;
  }

  delete[] fmsgs;
  delete[] bmsgs;
  delete[] buf;
  delete[] exp_node_pots;
  delete[] marginals;
  delete[] samples;

}
