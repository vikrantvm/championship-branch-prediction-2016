// -*- mode:c++; indent-tabs-mode:nil; -*-

// This source code is derived from ISL-TAGE (CBP-3).
// TAGE is based on the great work from Andre Seznec and Pierre Michaud.
// In this source code, my contribution for the performance is quite small.

// About ISL-TAGE, please refer to previous branch prediction championship.
// URL: http://www.jilp.org/jwac-2/

// About TAGE predictor, please refer to JILP online publication.
// URL: http://www.jilp.org/vol8/v8paper1.pdf

// In this predictor, we tried to combine local branch history to
// improve the performance of TAGE predictor because there are no
// effective way to exploit the local history for the partial tag
// matching. We combine global branch history and local branch
// history for the indexing part of TAGE branch predictor.
// It helps to reduce the branch miss prediction.

#ifndef _PREDICTOR_H_
#define _PREDICTOR_H_

#include "utils.h"
//#include "tracer.h"
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <inttypes.h>
#include <math.h>
#include <bitset>
#include <climits>
#include <vector>

// #define RESTRICT_VERSION

// #ifndef RESTRICT_VERSION
// // These feature do not affect the accuracy of the submitted branch predictor.
// // However, to confirm the temporary variables do not affect the overall
// // performance, these variables should be disabled.
//   #define FOLDEDINDEXING
//   #define NOTCLEARTEMPORARYVARIABLES
// #endif

// Enabling / disabling features...
// #define STATCOR
// #define LOOPPREDICTOR
// #define NEURAL
#define VOTING
#define TAGEV 14
// Table configuration parameters
#define NSTEP 4
#define NDIFF 3
#define NALLOC 5//4,5,6,7
#define NHIST 20
#define NSTAT 5
#define MSTAT 5//5,6,7
#define TSTAT (NSTAT+MSTAT+1)

// Local global branch history ratio
#define LSTEP 2
#define LG_RATIO 60

// History length settings
#define MINHIST 5
#define MAXHIST 880
#define MINCGHIST 2
#define MAXCGHIST 13
#define MINCLHIST 2
#define MAXCLHIST 12

// Tage parameters
#define HYSTSHIFT 2
#define LOGB 14
#define LOGG 12

// Statistic corrector parameters
#define LOGC 11
#define CBANK 3
#define CSTAT 6

// Loop predictor parameters
#define LOGL 4
#define LOOPWAY 3
#define LOOPTAG 10
#define LOOPITER 10
#define LOOPCONF 3
#define LOOPAGE 3

// Maximum history width
#define PHISTWIDTH 16

// LHT parameters
#define LHTBITS 6
#define LHTSIZE (1<<LHTBITS)
#define LHTMASK (LHTSIZE-1)
#define LHISTWIDTH ((MAXCLHIST>(MAXHIST/LG_RATIO))?MAXCLHIST:(MAXHIST/LG_RATIO))

// Misc counter width
#define DC_WIDTH 10
#define WL_WIDTH 7
#define UC_WIDTH 5
#define UT_WIDTH 6
#define UA_WIDTH 4
#define TK_WIDTH 8


//definitions for the neural predictor
//
#define PHT_CTR_MAX  3
#define PHT_CTR_INIT 2

#define HIST_LEN   17


#define WT_SIZE 10 // number of entries in weight tables
#define GHL 128	// length of global history for SGHR, GHR, HA and HTrain
#define WL_1 20 // history length for the separate 1024-entry tables
#define WL_2 16 // history length for the single 1024-entry table
#define WL_3 29 // history length for the single 512-entry table
#define THH 107 // Initial threshold of trainning


//////////////////////////////////////////////////////
// Base neural predictor class
//////////////////////////////////////////////////////

class NeuralPredictor{

private:
    /* --- weight tables --- */
    int8_t     weightT[1<<(WT_SIZE)][WL_1]; // Taken weight table for the most recent 20 branches. 7 * 1024 * 20 = 143,360 bits
    int8_t	   weightNT[1<<(WT_SIZE)][WL_1]; // Not-Taken weight table for the most recent 20 branches. 7 * 1024 * 20 = 143,360 bits
    int8_t	   weight1[1<<WT_SIZE][WL_2]; // Single weight table for the next 16 branches. 7 * 1024 * 16 = 114,688 bits
    int8_t	   weight2[1<<WT_SIZE][WL_3]; // Single weight table for the last 29 branches. 7 * 512 * 29 = 103,936 bits
    // Cost of weight tables: 519,680 bits (61.7KB)
    // Total Cost including misc info: 523,032 bits (62.1KB)

    /* --- Shift registers --- */
    bool *SGHR; //speculative history register used for updates
    bool *HTrain; // A register indicate if training is needed
    bool *GHR;	//perfect history used for prediction and updates
    uint64_t *HA;	// Path address register

    int8_t specul;  // Speculation counter. Counting how many branches are fetched but not retired.
    int32_t threshold; // dynamic threshold value as in O-GEHL
    int8_t TC; //threshold counter as in O-GEHL
    int32_t accum_g=0;
    bool pred_t = false;

public:
    void initNeuralPredictor(){
        // cout<<"did call init"<<endl;
        SGHR = new bool[GHL];
        HTrain = new bool[GHL];
        GHR = new bool[GHL];
        HA = new uint64_t[GHL];
        assert(SGHR);
        assert(HTrain);
        assert(GHR);
        assert(HA);

        for (int i = 0; i < (1<<WT_SIZE); i++) {
            for(int j = 0; j < WL_1; j++)
            {
                weightT[i][j] = 0;
                weightNT[i][j] = 0;
            }
            for(int j = 0; j < WL_2; j++)
            {
                weight1[i][j] = 0;
            }
        }

        for (int i = 0; i < (1<<(WT_SIZE-1)); i++) {
            for(int j = 0; j < WL_3; j++)
            {
                weight2[i][j] = 0;
            }
        }
        specul = 0;
        threshold = THH;
        TC = 0;
    }
    bool getNeuralPredictor(UINT64 PC){
        /*----- Algorithm of Prediction -----*/
        int32_t accum = 0;
        /*----- First WL_1 branches: use 1024-entry separate T/NT weight tables -----*/
        for(int j=0; j <WL_1; j++) {
            uint64_t widx = gen_widx(PC, HA[j], WT_SIZE); // compute index to access either weight table

            if( GHR[j] == 1)		// If history is Taken
                accum += weightT[widx][j];  // Then add Taken weight
            else 				// If history is Not-Taken
                accum += weightNT[widx][j]; // Then add Not-Taken weight
        }
        /*----- Next WL_2 branches: use 1024-enrty single weight tables -----*/
        for(int j=0; j<WL_2; j++)
        {
            uint64_t widx = gen_widx(PC, HA[WL_1+j], WT_SIZE); // compute index to access either weight table
            if( GHR[WL_1+j] == 1)
                accum += weight1[widx][j];
            else
                accum -= weight1[widx][j];
        }
        /*----- Last WL_3 branches: use 512-entry single weight tables -----*/
        for(int j=0; j<WL_3; j++)
        {
            uint64_t widx = gen_widx(PC, HA[WL_1+WL_2+j], WT_SIZE-1); // compute index to access weight table
            if( GHR[WL_1+WL_2+j] == 1)
                accum += weight2[widx][j];
            else
                accum -= weight2[widx][j];
        }
        bool pred = (accum >= 0); // Predict Taken if sum >= 0; Predict Not-Taken if sum <= 0;

        accum_g = accum;
        pred_t = pred;
        return pred;
    }
    void updateNeuralPredictor(UINT64 PC, OpType opType, bool resolveDir){
        // cout<<"did call update"<<endl;
        for(int j=GHL-1; j>0; j--)
        {
            SGHR[j] = SGHR[j-1];
            GHR[j] = GHR[j-1];
            HTrain[j] = HTrain[j-1];
            HA[j] = HA[j-1];
        }
        SGHR[0] = pred_t;
        // SGHR[0] = predDir;
        GHR[0] = resolveDir?1:0;
        if(accum_g>-threshold && accum_g<threshold)
            HTrain[0]=1; // 1 means trainning needed
        else
            HTrain[0]=0;
        HA[0] = PC;// HA records the path address
        specul++;	// Increment the speculation counter because a branch is fetched into the pipeline

        if ( opType == OPTYPE_CALL_DIRECT_COND || OPTYPE_JMP_DIRECT_COND) {
            bool t = resolveDir;
            if( (t != SGHR[specul-1]) || (HTrain[specul-1] == 1) ) 	//Training needed if threshold not exceeded or predict wrong
            {
                /*----- Algorithm for Update -----*/

                /*----- First WL_1 branches: update T/NT weight tables separately -----*/
                for(int j = 0; j < WL_1; j++)	{
                    uint64_t widx = gen_widx(PC, HA[j+specul], WT_SIZE); // compute the index to access either weight table;
                    if(t==1 && GHR[specul+j]==1)
                    { if(weightT[widx][j]<63) weightT[widx][j]++;}
                    else if(t==0 && GHR[specul+j]==1)
                    { if(weightT[widx][j]>-64) weightT[widx][j]--;}
                    else if(t==1 && GHR[specul+j]==0)
                    { if(weightNT[widx][j]<63) weightNT[widx][j]++;}
                    else if(t==0 && GHR[specul+j]==0)
                    { if(weightNT[widx][j]>-64) weightNT[widx][j]--;}
                }
                /*----- Next WL_2 branches: update regular weight tables -----*/
                for(int j = 0; j < WL_2; j++)	{
                    uint64_t widx = gen_widx(PC, HA[j+WL_1+specul], WT_SIZE); // compute the index to access either weight table;
                    if(t==GHR[specul+j+WL_1])
                    { if(weight1[widx][j]<63) weight1[widx][j]++;}
                    else
                    { if(weight1[widx][j]>-64) weight1[widx][j]--;}
                }
                /*----- Last WL_3 branches: update regular weight tables -----*/
                for(int j = 0; j < WL_3; j++)	{
                    uint64_t widx = gen_widx(PC, HA[j+WL_1+WL_2+specul], WT_SIZE-1); // compute the index to access either weight table;
                    if(t==GHR[specul+j+WL_1+WL_2])
                    { if(weight2[widx][j]<63) weight2[widx][j]++;}
                    else
                    { if(weight2[widx][j]>-64) weight2[widx][j]--;}
                }
            }
            /*------Update the threshold -------*/
            if(t != SGHR[specul-1]) {
                TC++;
                if(TC==63) {
                    TC = 0;
                    threshold++;
                }
            }
            else if(t==SGHR[specul-1] && HTrain[specul-1] == 1) {
                TC--;
                if(TC==-63) {
                    TC = 0;
                    threshold--;
                }
            }

            specul--; // Decrement speculation counter because a branch is retired.
        }
    }

    /*----- Hash function that calculating index of weight tables -----*/
    uint64_t gen_widx(uint64_t cur_pc, uint64_t path_pc, uint64_t wt_size){
        cur_pc = (cur_pc ) ^ (cur_pc / (1<<wt_size));
        path_pc = (path_pc) ^ (path_pc / (1<<wt_size));
        uint64_t widx = cur_pc ^ (path_pc);
        widx = widx % (1<<wt_size);
        return widx;
    }
};

//////////////////////////////////////////////////////
// Base counter class
//////////////////////////////////////////////////////

// Counter base class
template<typename T, int MAX, int MIN>
class Counter {
private:
  T ctr;
public:
  T read(T val=0) { return ctr+val; }
  bool pred() { return ctr >= 0; }

  bool satmax(){ return ctr == MAX; }
  bool satmin(){ return ctr == MIN; }

  void setmax(){ ctr = MAX; }
  void setmin(){ ctr = MIN; }

  void write(T v) {
    assert(v <= MAX);
    assert(v >= MIN);
    ctr = v;
  }

  void add(T d) {
    ctr = ctr + d;
    if (ctr > MAX){
      ctr = MAX;
    }else if (ctr < MIN){
      ctr = MIN;
    }
  }

  void update(bool incr) {
    if (incr) {
      if (ctr < MAX)
        ctr = ctr + 1;
    } else {
      if (ctr > MIN)
        ctr = ctr - 1;
    }
  }

  virtual int budget() = 0;
};

//signed integer counter
template<int WIDTH>
class SCounter : public Counter<int32_t,((1<<(WIDTH-1))-1),(-(1<<(WIDTH-1)))>{
public:
  virtual int budget() {return WIDTH;}
};

//unsigned integer counter
template<int WIDTH>
class UCounter : public Counter<int32_t,((1<<(WIDTH))-1),0>{
public:
  virtual int budget() {return WIDTH;}
};


//////////////////////////////////////////////////////
// history managemet data structure
//////////////////////////////////////////////////////

class GlobalHistoryBuffer {
  // This implementation is used to save the simulation time.
  static const int HISTBUFFERLENGTH = 4096;
private:
  int ptr;
  bool bhr[HISTBUFFERLENGTH];

public:
  int init() {
    for(int i=0; i<HISTBUFFERLENGTH; i++) { bhr[i] = false; }
    return MAXHIST;
  }

  void push(bool taken) {
    ptr--;
    // cout<<"ptr: "<<ptr<<endl;
    // cout<<"anding in "<<(ptr & (HISTBUFFERLENGTH-1))<<endl;
    bhr[ptr & (HISTBUFFERLENGTH-1)] = taken;
    // cout<<"pushing"<<endl;
    //for(int i=0; i<HISTBUFFERLENGTH; i++) { cout<<bhr[i]; }
    // cout<<endl;
  }

  // read n_th history
  bool read(int n) { return bhr[(n+ptr) & (HISTBUFFERLENGTH-1)]; }
};


class GlobalHistory : public GlobalHistoryBuffer {

private:
  // Folded index (this register save the hash value of the global history,
  // this values can be regenerated if the all branch histories are stored in the GHR)
  class FoldedHistory {
  public:
    unsigned comp;
    int CLENGTH;
    int OLENGTH;
    int OUTPOINT;

    void init (int original_length, int compressed_length) {
      comp = 0;
      OLENGTH = original_length;
      CLENGTH = compressed_length;
      OUTPOINT = OLENGTH % CLENGTH;
    }

    void update (GlobalHistoryBuffer *h) {
      // cout<<"here"<<endl;
      comp = (comp << 1) | h->read(0);
      comp ^= (h->read(OLENGTH) ? 1 : 0) << OUTPOINT;
      comp ^= (comp >> CLENGTH);
      comp &= (1 << CLENGTH) - 1;

    }
  };
  FoldedHistory ch_i[NHIST];
  FoldedHistory ch_c[NSTAT];
  FoldedHistory ch_t[3][NHIST];

public:
  void updateFoldedHistory() {
    // cout<<"then came here"<<endl;
    for (int i=0; i<NSTAT; i++) {
      ch_c[i].update(this);
    }
    for (int i = 0; i < NHIST; i++) {
      ch_i[i].update(this);
      ch_t[0][i].update(this);
      ch_t[1][i].update(this);
      ch_t[2][i].update(this);
    }
  }

  void setup(int *m, int *l, int *t, int *c, int size) {
    for (int i = 0; i < NHIST; i++) {
      ch_i[i].init(m[i], l[i]);
      //cout<<"ch_i_"<<i<<": "<<m[i]<<" "<<l[i]<<endl;
      ch_t[0][i].init(m[i], t[i]);
      //cout<<"ch_t_0_"<<i<<": "<<m[i]<<" "<<t[i]<<endl;
      ch_t[1][i].init(m[i], t[i] - 1);
      //cout<<"ch_t_1_"<<i<<": "<<m[i]<<" "<<t[i]-1<<endl;
      ch_t[2][i].init(m[i], t[i] - 2);
      //cout<<"ch_t_2_"<<i<<": "<<m[i]<<" "<<t[i]-2<<endl;
    }
    for (int i=0; i<NSTAT; i++) {
      ch_c[i].init(c[i], size);
    }
  }

  uint32_t gidx(int n, int length, int clength) {
    return ch_i[n].comp;
  }

  uint32_t gtag(int n, int length, int clength) {
    return ch_t[0][n].comp^(ch_t[1][n].comp<<1)^(ch_t[2][n].comp<<2);
  }

  uint32_t cgidx(int n, int length, int clength) {
    return ch_c[n].comp;
  }

public:
  void update(bool taken) {
    // cout<<"here";
    push(taken);
    updateFoldedHistory();
  }
};


class LocalHistory {
  uint32_t lht[LHTSIZE];

  uint32_t getIndex(uint32_t pc) {
    pc = pc ^ (pc >> LHTBITS) ^ (pc >> (2*LHTBITS));
    return pc & (LHTSIZE-1);
  }

public:
  int init() {
    for(int i=0; i<LHTSIZE; i++) {
      lht[i] = 0;
    }
    return LHISTWIDTH * LHTSIZE;
  }

  void update(uint32_t pc, bool taken) {
    lht[getIndex(pc)] <<= 1;
    lht[getIndex(pc)] |= taken ? 1 : 0;
    lht[getIndex(pc)] &= (1<<LHISTWIDTH) - 1;
  }

  uint32_t read(uint32_t pc, int length, int clength) {
    uint32_t h = lht[getIndex(pc)];
    h &= (1 << length) - 1;

    uint32_t v = 0;
    while(length > 0) {
      v ^= h;
      h >>= clength;
      length -= clength;
    }
    return v & ((1 << clength) - 1);
  }
};

//////////////////////////////////////////////////////////
// Base predictor for TAGE predictor
// This predictor is derived from CBP3 ISL-TAGE
//////////////////////////////////////////////////////////

template <int BITS, int HSFT>
class Bimodal {
private:
  bool pred[1 << BITS];
  bool hyst[1 << (BITS-HSFT)];
  uint32_t getIndex(uint32_t pc, int shift=0) {
    return (pc & ((1 << BITS)-1)) >> shift ;
  }

public:
  int init() {
    for(int i=0; i<(1<<BITS); i++) { pred[i] = 0; }
    for(int i=0; i<(1<<(BITS-HSFT)); i++) { hyst[i] = 1; }
    return (1<<BITS)+(1<<(BITS-HSFT));
  }

  bool predict(uint32_t pc) {
    return pred[getIndex(pc)];
  }

  void update(uint32_t pc, bool taken) {
    int inter = (pred[getIndex(pc)] << 1) + hyst[getIndex(pc, HSFT)];
    // cout<<"pred"<<getIndex(pc)<<endl;
    // cout<<"hyst"<<getIndex(pc, HSFT)<<endl;
    if(taken) {
      if (inter < 3) { inter++; }
    } else {
      if (inter > 0) { inter--; }
    }
    pred[getIndex(pc)] = (inter >= 2);
    hyst[getIndex(pc,HSFT)] = ((inter & 1)==1);
  }
};

//////////////////////////////////////////////////////////
// Global component for TAGE predictor
// This predictor is derived from CBP3 ISL-TAGE
//////////////////////////////////////////////////////////

class GEntry {
public:
  uint32_t tag;
  SCounter<3> c;
  UCounter<2> u;

  GEntry () {
    tag = 0;
    c.write(0);
    u.write(0);
  }

  void init(uint32_t t, bool taken, int uval=0) {
    tag = t;
    c.write(taken ? 0 : -1);
    u.write(uval);
  }

  bool newalloc() {
    return (abs(2*c.read() + 1) == 1);
  }
};

//////////////////////////////////////////////////////////
// Put it all together.
// The predictor main component class
//////////////////////////////////////////////////////////

// Configuration of table sharing strategy
static const int STEP[NSTEP+1] = {0, NDIFF, NHIST/2, NHIST-NDIFF, NHIST};

class my_predictor {

  /////////////////////////////////////////////////////////
  // Constant Values
  //
  // These variables are not changed during simulation...
  // So, we do not count these variables as the storage.
  /////////////////////////////////////////////////////////

  // Tag width and index width of TAGE predictor
  int TB[NHIST];
  int logg[NHIST];

  // History length for TAGE predictor
  int m[NHIST];
  int l[NHIST];
  int p[NHIST];

  // History length for statistical corrector predictors
  int cg[NSTAT];
  int cp[NSTAT];
  int cl[MSTAT];

  // // Table for TAGE Voter
  // int voter_weight[NHIST][1<<LOGG];
  // //init with INT_MIN value for undefined state.
  // // for(int i = 0; i < NHIST; ++i){
  // //   for(int j = 0; j < (1<<LOGG); ++j)
  // //     voter_weight[i][j] = 0;//INT_MIN;
  // // }
  // int64_t VoterIndex;
  // vector<int> VoterTagMatch;
  // //HA = new uint64_t[4096];
  // // bool *SGHR;
  // // bool *HTrain;
  // // SGHR = new bool[4096];
  // // HTrain = new bool[4096];
  // bool SGHR[4096];
  // bool HTrain[4096];
  // int sum_wt_g;
  // int threshold = 100;
  // int8_t TC;
  // Table for TAGE Voter
  int voter_weight[1<<TAGEV][NHIST];
  // int voter_weight[1<<LOGG][NHIST];
  //init with INT_MIN value for undefined state.
  int64_t VoterIndex;
  vector<int> VoterTagMatch;
  bool SGHR[GHL];
  bool HTrain[GHL];
  uint64_t HA[GHL];
  int sum_wt_g;
  int threshold;
  int8_t TC;


  /////////////////////////////////////////////////////////
  // Temporary values
  //
  // These values can be computed from the prediction resources,
  // but we use these values to save the simulation time
  /////////////////////////////////////////////////////////

  // Index variables of TAGE and some other predictors
  uint32_t CI[TSTAT];
  uint32_t GI[NHIST];
  uint32_t GTAG[NHIST];

  // Intermediate prediction result for TAGE
  bool HitPred, AltPred, TagePred;
  int HitBank, AltBank, TageBank;

  // Intermediate prediction result for statistical corrector predictor
  bool SCPred;
  int SCSum;

  // Intermediate prediction result for loop predictor
  bool loopPred;
  bool loopValid;

  // This function is used for confirming the reset of temporal values
  void clearTemporaryVariables() {
    memset(CI, 0, sizeof(CI));
    memset(GI, 0, sizeof(GI));
    memset(GTAG, 0, sizeof(GTAG));
    SCPred = HitPred = AltPred = TagePred = false;
    SCSum  = HitBank = AltBank = TageBank = 0;
    loopPred = loopValid = 0;
  }

  /////////////////////////////////////////////////////////
  // Hardware resoruce
  //
  // These variables are counted as the actual hardware
  // These variables are not exceed the allocated budgeet (32KB + 1K bit)
  /////////////////////////////////////////////////////////

  // Prediction Tables
  #ifdef NEURAL
  NeuralPredictor nPred;
  #endif
  Bimodal<LOGB,HYSTSHIFT> btable; // bimodal table
  GEntry *gtable[NHIST]; // global components
  SCounter<CSTAT> *ctable[2]; // statistical corrector predictor table
  // LoopPredictor ltable; // loop predictor

  // Branch Histories
  GlobalHistory ghist; // global history register
  LocalHistory lhist; // local history table
  uint32_t phist; // path history register

  // Profiling Counters
  SCounter<DC_WIDTH> DC; // difficulty counter
  SCounter<WL_WIDTH> WITHLOOP; // loop predictor usefulness
  UCounter<UC_WIDTH> UC; // statistical corrector predictor tracking counter
  SCounter<UT_WIDTH> UT; // statistical corrector predictor threshold counter
  UCounter<TK_WIDTH> TICK; // tick counter for reseting u bit of global entryies
  SCounter<UA_WIDTH> UA[NSTEP+1][NSTEP+1]; // newly allocated entry counter

private:

  //////////////////////////////////////////////////
  // Setup history length
  //////////////////////////////////////////////////

  void setHistoryLength(int *len, int num, int min, int max) {
    for(int i=0; i<num; i++) {
      double a = (double)max / (double)min;
      double j = (double)i/(double)(num-1);
      len[i] = (int)((min * pow(a, j)) + 0.5);
    }
    assert(len[0] == min);
    assert(len[num-1] == max);
  }

  void setupHistoryConfiguration() {
    printf("Setup history length...\n");
    setHistoryLength(m, NHIST, MINHIST, MAXHIST);
    setHistoryLength(cg, NSTAT, MINCGHIST, MAXCGHIST);
    setHistoryLength(cl, MSTAT, MINCLHIST, MAXCLHIST);

    for (int i=0; i<NHIST; i++) {
      l[i] = (int)(m[i] / LG_RATIO);
      l[i] = (l[i] > LHISTWIDTH) ? LHISTWIDTH : l[i];
      if(i < STEP[LSTEP]) {
        l[i] = 0;
      }
    }

    cg[0] -= 2;
    cg[2] -= 1;
    cg[3] += 3;
    cg[4] -= 13;
    cl[1] -= 2;
    m[0] -= 5;
    m[1] -= 3;
    m[2] -= 1;
    l[10] -= 1;
    l[11] += 1;
    l[14] -= 1;

    for (int i=0; i<NHIST; i++) {
      p[i] = m[i];
      p[i] = (p[i] > PHISTWIDTH) ? PHISTWIDTH : p[i];
    }
    printf("\n");
    for (int i=0; i<NSTAT; i++) {
      cp[i] = (cg[i] > PHISTWIDTH) ? PHISTWIDTH : cg[i];
    }

    for (int i=0; i<NHIST; i++) {
      if(i > 0) {
        assert(m[i-1] <= m[i]);
        assert(p[i-1] <= p[i]);
        assert(l[i-1] <= l[i]);
      }
      assert(m[i] >= 0);
      assert(m[i] <= MAXHIST);
      assert(p[i] >= 0);
      assert(p[i] <= PHISTWIDTH);
      assert(l[i] >= 0);
      assert(l[i] <= LHISTWIDTH);
    }
    for (int i=0; i<NSTAT; i++) {
      assert(cg[i] >= 0);
      assert(cp[i] >= 0);
      assert(cg[i] <= MAXHIST);
      assert(cp[i] <= MAXHIST);
    }
    for (int i=0; i<MSTAT; i++) {
      assert(cl[i] >= 0);
      assert(cl[i] <= LHISTWIDTH);
    }
    for(int i=0; i<NHIST; i++) printf("m[%d] = %d, l[%d] = %d\n", i, m[i], i, l[i]);
    for(int i=0; i<NSTAT; i++) printf("cg[%d] = %d\n", i, cg[i]);
    for(int i=0; i<MSTAT; i++) printf("cl[%d] = %d\n", i, cl[i]);
  }

public:

  my_predictor (void) {
    int budget=0;

    // Setup history length
    setupHistoryConfiguration();

    // Setup misc registers
    WITHLOOP.write(-1);
    DC.write(0);
    UC.write(0);
    UT.write(0);
    TICK.write(0);
    budget += WITHLOOP.budget();
    budget += DC.budget();
    budget += UC.budget();
    budget += UT.budget();
    budget += TICK.budget();

    for(int i=0; i<NSTEP+1; i++) {
      for(int j=0; j<NSTEP+1; j++) {
        UA[i][j].write(0);
        budget += UA[i][j].budget();
      }
    }

    // Setup global components
    logg[STEP[0]] = LOGG + 1;
    logg[STEP[1]] = LOGG + 3;
    logg[STEP[2]] = LOGG + 2;
    logg[STEP[3]] = LOGG - 1;
    TB[STEP[0]] =  7;
    TB[STEP[1]] =  9;
    TB[STEP[2]] = 11;
    TB[STEP[3]] = 13;
    for(int i=0; i<NSTEP; i++) {
      gtable[STEP[i]] = new GEntry[1 << logg[STEP[i]]];
      budget += (2/*U*/+3/*C*/+TB[STEP[i]]) * (1<<logg[STEP[i]]);
    }
    for(int i=0; i<NSTEP; i++) {
      for (int j=STEP[i]+1; j<STEP[i+1]; j++) {
        logg[j]=logg[STEP[i]]-3;
        gtable[j] = gtable[STEP[i]];
        TB[j] = TB[STEP[i]];
      }
    }

    // Setup bimodal table
    budget += btable.init();

    // Setup statistic corrector predictor
    ctable[0] = new SCounter<CSTAT>[1 << LOGC];
    ctable[1] = new SCounter<CSTAT>[1 << LOGC];
    for(int i=0; i<(1<<LOGC); i++) {
      ctable[0][i].write(-(i&1));
      ctable[1][i].write(-(i&1));
    }
    budget += 2 * CSTAT * (1<<LOGC);

    // Setup loop predictor
    // budget += ltable.init();

    // Setup history register & table
    phist = 0;
    ghist.init();
    lhist.init();
    ghist.setup(m, logg, TB, cg, LOGC-CBANK);
    budget += PHISTWIDTH;
    budget += m[NHIST-1];
    budget += LHISTWIDTH * LHTSIZE;

    sum_wt_g=0;
    threshold = 75;
    TC = 0;
  //set TAGE voting weights to zero
  for(int i = 0; i < (1<<TAGEV); ++i){
    for(int j = 0; j < NHIST; ++j)
      voter_weight[i][j] = 0;//INT_MIN;
  }

    #ifdef NEURAL
    nPred.initNeuralPredictor();
    #endif
    // Output the total hardware budget
    printf("Total Budget: Limit:%d, %d %d\n", (32*1024*8+1024), budget, budget/8);
  }


  //////////////////////////////////////////////////////////////
  // Hash functions for TAGE and static corrector predictor
  //////////////////////////////////////////////////////////////

  int F (int A, int size, int bank, int width) {
    int A1, A2;
    int rot = (bank+1) % width;
    A = A & ((1 << size) - 1);
    A1 = (A & ((1 << width) - 1));
    A2 = (A >> width);
    A2 = ((A2 << rot) & ((1 << width) - 1)) + (A2 >> (width - rot));
    A = A1 ^ A2;
    A = ((A << rot) & ((1 << width) - 1)) + (A >> (width - rot));
    return (A);
  }

  // gindex computes a full hash of pc, ghist and phist
  uint32_t gindex(uint32_t pc, int bank, int hist) {
    // we combine local branch history for the TAGE index computation
    uint32_t index =
      lhist.read(pc, l[bank], logg[bank]) ^
      ghist.gidx(bank, m[bank], logg[bank]) ^
      F(hist, p[bank], bank, logg[bank]) ^
      (pc >> (abs (logg[bank] - bank) + 1)) ^ pc ;
    return index & ((1 << logg[bank]) - 1);
  }

  //  tag computation for TAGE predictor
  uint32_t gtag(uint32_t pc, int bank) {
    uint32_t tag = ghist.gtag(bank, m[bank], TB[bank]) ^ pc ;
    return (tag & ((1 << TB[bank]) - 1));
  }

  // index computation for statistical corrector predictor
  uint32_t cgindex (uint32_t pc, int bank, int hist, int size) {
    uint32_t index =
      ghist.cgidx(bank, cg[bank], size) ^
      F(hist, cp[bank], bank, size) ^
      (pc >> (abs (size - (bank+1)) + 1)) ^ pc ;
    return index & ((1 << size) - 1);
  }

  // index computation for statistical corrector predictor
  uint32_t clindex (uint32_t pc, int bank, int size) {
    uint32_t index =
      lhist.read(pc, cl[bank], size) ^
      (pc >> (abs (size - (bank+1)) + 1)) ^ pc ;
    return index & ((1 << size) - 1);
  }

  // index computation for usefulness of AltPred counters
  uint32_t uaindex(int bank) {
    for(int i=0; i<NSTEP; i++) {
      if(bank < STEP[i]) return i;
    }
    return NSTEP;
  }

  uint64_t gen_widx(uint64_t cur_pc, uint64_t path_pc, uint64_t wt_size, int bank){
      cur_pc = (cur_pc ) ^ (cur_pc / (1<<wt_size));
      path_pc = (path_pc) ^ (path_pc / (1<<wt_size));
      uint64_t widx = cur_pc ^ (path_pc) ^ ghist.gidx(bank, m[bank], logg[bank]);
      widx = widx % (1<<wt_size);
      return widx;
  }


  //////////////////////////////////////////////////////////////
  // Actual branch prediction and training algorithm
  //////////////////////////////////////////////////////////////

  //compute the prediction
  bool predict(uint32_t pc, uint16_t brtype) {
    // Final prediction result
    bool pred_taken = true;

    if (brtype == OPTYPE_CALL_DIRECT_COND || OPTYPE_JMP_DIRECT_COND/*OPTYPE_BRANCH_COND*/) {
#ifndef NOTCLEARTEMPORARYVARIABLES
      clearTemporaryVariables();
#endif

      // Compute index values
      for (int i = 0; i < NHIST; i++) {
        GI[i] = gindex(pc, i, phist);
        GTAG[i] = gtag(pc, i);
      }

      // Update the index values for interleaving
      for (int s=0; s<NSTEP; s++) {
        for (int i=STEP[s]+1; i<STEP[s+1]; i++) {
          GI[i]=((GI[STEP[s]]&7)^(i-STEP[s]))+(GI[i]<<3);
        }
      }

      // Compute the prediction result of TAGE predictor
      HitBank = AltBank = -1;
      #ifdef NEURAL
      HitPred = AltPred = nPred.getNeuralPredictor(pc);
      #else
      HitPred = AltPred = btable.predict(pc);
      #endif

      //original TAGE
      for (int i=0; i<NHIST; i++) {
        if (gtable[i][GI[i]].tag == GTAG[i]) {
          AltBank = HitBank;
          HitBank = i;
          AltPred = HitPred;
          HitPred = gtable[i][GI[i]].c.pred();
          VoterTagMatch.push_back(i);
        }
      }

      // Select the highest confident prediction result
      //disable when using voting
      TageBank = HitBank;
      TagePred = HitPred;
      int sum_weights = 0;

      #ifdef VOTING
      if(HitBank != -1){
      int sum_weights = 0;
      for (int i = 0; i < VoterTagMatch.size(); i++) {
        VoterIndex = gen_widx(pc, VoterTagMatch[i], TAGEV, VoterTagMatch[i]);//, VoterTagMatch[i]);
        if(gtable[VoterTagMatch[i]][GI[VoterTagMatch[i]]].c.pred()){
          sum_weights += voter_weight[VoterIndex][VoterTagMatch[i]]*1;
        }
        else{
          sum_weights += voter_weight[VoterIndex][VoterTagMatch[i]]*-1;
        }
      }
      if(sum_weights>0){
        pred_taken=true;
      }
      else if(sum_weights<0){
        pred_taken=false;
      }
      else{
        pred_taken=HitPred;
      }

      sum_wt_g = sum_weights;
      for(int j=GHL-1; j>0; j--)
      {
          SGHR[j] = SGHR[j-1];
          HTrain[j] = HTrain[j-1];
          HA[j] = HA[j-1];
      }
      SGHR[0] = pred_taken;
      }else{
      pred_taken = HitPred;
      }
      #endif


      // if (HitBank >= 0) {
      //   int u = UA[uaindex(HitBank)][uaindex(AltBank)].read();
      //   if((u>=0)&&gtable[HitBank][GI[HitBank]].newalloc()) {
      //     TagePred = AltPred;
      //     TageBank = AltBank;
      //   }
      // }

      #ifndef VOTING
      pred_taken = TagePred;
      #endif
    }



    return pred_taken;
  }

  // void update(uint32_t pc, uint16_t brtype, bool taken, uint32_t target) {
     void update(uint32_t pc, OpType brtype, bool taken, uint32_t target) {
    if (brtype == OPTYPE_CALL_DIRECT_COND || OPTYPE_JMP_DIRECT_COND/*OPTYPE_BRANCH_COND*/) {
#ifndef NOTCLEARTEMPORARYVARIABLES
      clearTemporaryVariables();
      predict(pc, brtype); // re-calculate temporal variables
#endif

      // Determining the allocation of new entries
      bool ALLOC = (TagePred != taken) && (HitBank < (NHIST-1));
      if (HitBank >= 0) {
        if (gtable[HitBank][GI[HitBank]].newalloc()) {
          if (HitPred == taken) {
            ALLOC = false;
          }
          if (HitPred != AltPred) {
            UA[uaindex(HitBank)][uaindex(AltBank)].update(AltPred == taken);
          }
        }
      }


      if (ALLOC) {
        // Allocate new entries up to "NALLOC" entries are allocated
        int T = 0;
        for (int i=HitBank+1; i<NHIST; i+=1) {
          if (gtable[i][GI[i]].u.read() == 0) {
            gtable[i][GI[i]].init(GTAG[i], taken, 0);
            TICK.add(-1);
            if (T == NALLOC) break;
            T += 1;
            i += 1 + T/2; // After T th allocation, we skip 1+(T/2) tables.
          } else {
            TICK.add(+1);
          }
        }

        // Reset useful bit to release OLD useful entries
        // When the accuracy of the predictor is lower than pre-defined
        // threshold, we aggressively reset the useful counters.
        bool resetUbit = ((T == 0) && DC.pred()) || TICK.satmax();
        if (resetUbit) {
          TICK.write(0);
          for (int s=0; s<NSTEP; s++) {
            for (int j=0; j<(1<<logg[STEP[s]]); j++)
              gtable[STEP[s]][j].u.add(-1);
          }
        }
      }

      // Tracking prediction difficulty of the current workload
      // This counter represent the prediction accuracy of the
      // TAGE branch predictor.
      DC.add((TagePred == taken) ? -1 : 32);

      // Update prediction tables
      // This part is same with ISL-TAGE branch predictor.
      if (HitBank >= 0) {
        gtable[HitBank][GI[HitBank]].c.update(taken);
        if ((gtable[HitBank][GI[HitBank]].u.read() == 0)) {
          if (AltBank >= 0) {
            gtable[AltBank][GI[AltBank]].c.update(taken);
          } else {
            #ifdef NEURAL
            nPred.updateNeuralPredictor(pc, brtype, taken);
            #else
            btable.update(pc, taken);
            #endif
          }
        }
      } else {
        #ifdef NEURAL
        nPred.updateNeuralPredictor(pc, brtype, taken);
        #else
        btable.update(pc, taken);
        #endif
      }
      //update neural predictor
      #ifdef VOTING
      bool t = taken;
      if(sum_wt_g>-threshold && sum_wt_g<threshold)
          HTrain[0]=1; // 1 means training needed
      else
          HTrain[0]=0;
      if(HitBank != -1){
      if( (t != SGHR[0]) || (HTrain[0] == 1) ) {
        for (uint i = 0; i < VoterTagMatch.size(); i++) {
          VoterIndex = gen_widx(pc, VoterTagMatch[i], TAGEV, VoterTagMatch[i]);//, VoterTagMatch[i]);
          if(t==1 && gtable[VoterTagMatch[i]][GI[VoterTagMatch[i]]].c.pred() == true){

            if(voter_weight[VoterIndex][VoterTagMatch[i]]<63){
              voter_weight[VoterIndex][VoterTagMatch[i]]++;
            }
            }
          else if(t==0 && gtable[VoterTagMatch[i]][GI[VoterTagMatch[i]]].c.pred() == true){
            if(voter_weight[VoterIndex][VoterTagMatch[i]]>-64){
              voter_weight[VoterIndex][VoterTagMatch[i]]--;
            }
            }
          else if(t==1 && gtable[VoterTagMatch[i]][GI[VoterTagMatch[i]]].c.pred() == false){
            if(voter_weight[VoterIndex][VoterTagMatch[i]]>-64){
              voter_weight[VoterIndex][VoterTagMatch[i]]--;
            }
            }
          else if(t==0 && gtable[VoterTagMatch[i]][GI[VoterTagMatch[i]]].c.pred() == false){
            if(voter_weight[VoterIndex][VoterTagMatch[i]]<63){
              voter_weight[VoterIndex][VoterTagMatch[i]]++;
            }
            }
          }
      }

      if(t != SGHR[0]) {
          TC++;
          if(TC==63) {
              TC = 0;
              threshold++;
          }
      }
      else if(t==SGHR[0] && HTrain[0] == 1) {
          TC--;
          if(TC==-64) {
              TC = 0;
              threshold--;
          }
      }
    }
    VoterTagMatch.clear();
    HA[0]=pc;
    #endif


      // Update useful bit counter
      // This useful counter updating strategy is derived from
      // Re-reference interval prediction.
      if (HitBank >= 0) {
        bool useful = (HitPred == taken) && (AltPred != taken) ;
        if(useful) {
          gtable[HitBank][GI[HitBank]].u.setmax();
        }
      }
    }

    //////////////////////////////////////////////////
    // Branch history management.
    //////////////////////////////////////////////////

    // How many history bits are inserted for the buffer
    int maxt = 1;

    // Special treamtment for function call
    if (brtype == OPTYPE_CALL_INDIRECT_COND/*OPTYPE_INDIRECT_BR_CALL*/) maxt = 3 ;
    if (brtype == OPTYPE_CALL_DIRECT_COND     ) maxt = 5 ;

    // Branch history information
    // (This feature is derived from ISL-TAGE)
    int T = ((target ^ (target >> 3) ^ pc) << 1) + taken;
    for (int t = 0; t < maxt; t++) {
      ghist.update((T >> t) & 1);
      phist <<= 1;
      phist += (pc >> t) & 1;
      phist &= (1 << PHISTWIDTH) - 1;

    }


    // Update local history
    lhist.update(pc, taken);
  }

};

/////////////////////////////////////////////////////////////

class PREDICTOR{

 private:
  my_predictor *tage;

 public:

  // The interface to the four functions below CAN NOT be changed

  PREDICTOR(void);
  bool    GetPrediction(UINT64 PC, bool btbANSF, bool btbATSF, bool btbDYN);
  void    UpdatePredictor(UINT64 PC, OpType opType, bool resolveDir, bool predDir, UINT64 branchTarget, bool btbANSF, bool btbATSF, bool btbDYN);
  void    TrackOtherInst(UINT64 PC, OpType opType, bool  branchDir, UINT64 branchTarget);

  // Contestants can define their own functions below

};

/////////////////////////////////////////////////////////////

#endif
