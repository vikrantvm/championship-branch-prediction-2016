#ifndef _PREDICTOR_H_
#define _PREDICTOR_H_
#define LOOPPREDICTOR
#define GLOCAL
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <inttypes.h>
#include <math.h>
#include "utils.h"

using namespace std;

// Loop predictor parameters
#define LOGL 10//4
#define LOOPWAY 3
#define LOOPTAG 10
#define LOOPITER 10
#define LOOPCONF 3
#define LOOPAGE 3
#define WL_WIDTH 7


// LHT parameters
#define LHTBITS 10//6
#define LHTSIZE (1<<LHTBITS)
#define LHTMASK (LHTSIZE-1)
#define LHISTWIDTH ((MAXCLHIST>(MAXHIST/LG_RATIO))?MAXCLHIST:(MAXHIST/LG_RATIO))

// Local global branch history ratio
#define LSTEP 2
#define LG_RATIO 20//60

#define NHIST 1

// History length settings
#define MINHIST 5
#define MAXHIST 128//880
#define MINCGHIST 2
#define MAXCGHIST 13
#define MINCLHIST 2
#define MAXCLHIST 12

#define WT_SIZE 16 // number of entries in weight tables
#define GHL 256	// length of global history for SGHR, GHR, HA and HTrain
#define WL_1 24
#define WL_2 24
#define WL_3 48
#define THH 107 // Initial threshold of trainning

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

template<int WIDTH>
class UCounter : public Counter<int32_t,((1<<(WIDTH))-1),0>{
public:
  virtual int budget() {return WIDTH;}
};

class LoopPredictorEntry {
public:
  bool dir; // 1-bit
  uint32_t TAG; // 10-bit
  UCounter<LOOPITER> NIter; // 10-bit
  UCounter<LOOPITER> CIter; // 10-bit
  UCounter<LOOPCONF> confid; // 3-bit
  UCounter<LOOPAGE> age; // 3-bit

  LoopPredictorEntry () {
    confid.write(0);
    CIter.write(0);
    NIter.write(0);
    age.write(0);
    TAG = 0;
    dir = false;
  }

  int init(uint32_t ltag=0, bool taken=0) {
    dir = !taken;
    TAG = ltag;
    NIter.write(0);;
    age.write(7);
    confid.write(0);
    CIter.write(0);

    int budget = 0;
    budget += 1; //dir
    budget += LOOPTAG;
    budget += NIter.budget();
    budget += CIter.budget();
    budget += confid.budget();
    budget += age.budget();

    return budget; // total budget size (37-bit)
  }

  // Generate prediction
  bool predict(bool &valid) {
    valid = confid.satmax();
    if((CIter.read() + 1) == NIter.read()) {
      return !dir;
    } else {
      return  dir;
    }
  }

  void update(bool taken, bool useful) {
    bool valid;
    bool predloop = predict(valid);

    // Update confidence level
    if (valid) {
      if (taken != predloop) {
        NIter.write(0);
        age.write(0);
        confid.write(0);
        CIter.write(0);
        return;
      } else if (useful) {
        age.add(+1);
      }
    }

    // Increase the loop count
    CIter.add(+1);
    if (CIter.satmax()) {
      confid.write(7);
    }

    // When the loop count perform overflow, the confidence level set to 0
    if (CIter.read() > NIter.read()) {
      confid.write(0);
      NIter.write(0);
    }

    // When the direction is different from "dir".
    // checked the loop exit part or not.
    if (taken != dir) {
      bool success = CIter.read() == NIter.read();
      if(success) confid.add(+1);
      else        confid.setmin();
      NIter.write(CIter.read());
      CIter.write(0);

      // For short loop, the loop predictor do not applied.
      if (NIter.read() < 3) {
        dir = taken;
        NIter.write(0);
        age.write(0);
        confid.write(0);
      }
    }
  }
};

class LoopPredictor {
  static const int LOOPSIZE = LOOPWAY * (1<<LOGL);
  LoopPredictorEntry table[LOOPSIZE];

  static const int SEEDWIDTH = 6;
  int seed;

protected:
  int randomValue() {
    seed++;
    seed &= (1 << SEEDWIDTH) - 1;
    return seed ^ (seed >> 3) ;
  }

  // Hash function for index and tag...
  uint32_t getIndex(uint32_t pc, int way) {
    uint32_t v0 = pc & ((1 << (LOGL)) - 1);
    uint32_t v1 = (pc >> LOGL) & ((1 << (LOGL)) - 1);
    return (v0 ^ (v1 >> way)) | (way << LOGL) ;
  }

  uint32_t getTag(uint32_t pc) {
    uint32_t t;
    t = (pc >> (LOGL)) & ((1 << 2 * LOOPTAG) - 1);
    t ^= (t >> LOOPTAG);
    t = (t & ((1 << LOOPTAG) - 1));
    return t;
  }

  // Searching hit entry
  int searchEntry(uint32_t pc, uint32_t ltag) {
    for (int i = 0; i < LOOPWAY; i++) {
      int index = getIndex(pc, i);
      if (table[index].TAG == ltag) {
        return index;
      }
    }
    return -1;
  }

public:

  // Budget counting
  int init() {
    int budget = 0;

    seed = 0;
    budget += SEEDWIDTH;

    for(int i=0; i<LOOPSIZE; i++) {
      budget += table[i].init();
    }
    return budget;
  }

  // Generate prediction
  bool predict (uint32_t pc, bool &valid) {
    int LHIT = searchEntry(pc, getTag(pc));
    if(LHIT >= 0) {
      return table[LHIT].predict(valid);
    }
    valid = false;
    return (false);
  }

  // Update predictor
  void update (uint32_t pc, bool taken, bool alloc, bool useful) {
    int LHIT = searchEntry(pc, getTag(pc));
    if (LHIT >= 0) {
      useful = useful || ((randomValue() & 7) == 0) ;
      table[LHIT].update(taken, useful);
    } else if (alloc) {
      if ((randomValue() & 3) == 0) {
        uint32_t X = randomValue();
        for (int i = 0; i < LOOPWAY; i++) {
          int index = getIndex(pc, (X + i) % LOOPWAY);
          if (table[index].age.read() == 0) {
            table[index].init(getTag(pc), taken);
            return;
          }
        }
        for (int i = 0; i < LOOPWAY; i++) {
          int index = getIndex(pc, i);
          table[index].age.add(-1);
        }
      }
    }
  }
};

//////////////////////////////////////////////////////
// history management data structure
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
    bhr[ptr & (HISTBUFFERLENGTH-1)] = taken;
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
      comp = (comp << 1) | h->read(0);
      comp ^= (h->read(OLENGTH) ? 1 : 0) << OUTPOINT;
      comp ^= (comp >> CLENGTH);
      comp &= (1 << CLENGTH) - 1;
    }
  };
  FoldedHistory ch_i[NHIST];

public:
  void updateFoldedHistory() {
    for (int i = 0; i < NHIST; i++) {
      ch_i[i].update(this);
    }
  }

  void setup(int m, int l) {
    for (int i = 0; i < NHIST; i++) {
      ch_i[i].init(m, l);
    }
  }

  uint32_t gidx(int n, int length, int clength) {
    return ch_i[n].comp;
  }

public:
  void update(bool taken) {
    push(taken);
    updateFoldedHistory();
  }
};

class LocalHistory {
  uint32_t lht[LHTSIZE];

  uint32_t getIndex(uint64_t pc) {
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

  void update(uint64_t pc, bool taken) {
    lht[getIndex(pc)] <<= 1;
    lht[getIndex(pc)] |= taken ? 1 : 0;
    lht[getIndex(pc)] &= (1<<LHISTWIDTH) - 1;
  }

  uint32_t read(uint64_t pc, int length, int clength) {
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

class NeuralPredictor{

private:
  //weight tables
  int8_t     weightT[1<<(WT_SIZE)][WL_1];
  int8_t	   weightNT[1<<(WT_SIZE)][WL_1];
  int8_t	   weight1[1<<WT_SIZE][WL_2];
  int8_t	   weight2[1<<WT_SIZE][WL_3];


  //Shift registers
  bool *SGHR; //speculative history register used for updates
  bool *HTrain; // A register indicate if training is needed
  bool *GHR;	//perfect history used for prediction and updates
  uint64_t *HA;	// Path address register

  int8_t specul;  // Speculation counter. Counting how many branches are fetched but not retired.
  int32_t threshold; // dynamic threshold value as in O-GEHL
  int8_t TC; //threshold counter as in O-GEHL
  int32_t accum_g=0;

  // Intermediate prediction result for loop predictor

public:
  LoopPredictor ltable;
  SCounter<WL_WIDTH> WITHLOOP; // loop predictor usefulness
  // Branch Histories
  GlobalHistory ghist; // global history register
  LocalHistory lhist; // local history table

  bool SCPred;
  int SCSum;

  bool loopPred=0;
  bool loopValid=0;

  void initNeuralPredictor(){
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
    ghist.init();
    lhist.init();
    ghist.setup(128, 16);
  }


  bool getNeuralPredictor(UINT64 PC){

    int32_t accum = 0;

    for(int j=0; j <WL_1; j++) {
      uint64_t widx = gen_widx(PC, HA[j], WT_SIZE); // compute index to access either weight table

      if(GHR[j] == 1)		// If history is Taken
      accum += weightT[widx][j];  // Then add Taken weight
      else 				// If history is Not-Taken
      accum += weightNT[widx][j]; // Then add Not-Taken weight
    }

    for(int j=0; j<WL_2; j++)
    {
      uint64_t widx = gen_widx(PC, HA[WL_1+j], WT_SIZE); // compute index to access either weight table
      if( GHR[WL_1+j] == 1)
      accum += weight1[widx][j];
      else
      accum -= weight1[widx][j];
    }

    for(int j=0; j<WL_3; j++)
    {
      uint64_t widx = gen_widx(PC, HA[WL_1+WL_2+j], WT_SIZE-1); // compute index to access weight table
      if( GHR[WL_1+WL_2+j] == 1)
      accum += weight2[widx][j];
      else
      accum -= weight2[widx][j];
    }


    bool pred = (accum >= 0); // Predict Taken if sum >= 0; Predict Not-Taken if sum <= 0;

    #ifdef LOOPPREDICTOR
    loopPred = ltable.predict (PC, loopValid);
    if((WITHLOOP.pred()) && (loopValid)) {
      pred = loopPred ;
    }
    #endif

    accum_g = accum;
    return pred;
  }
  void updateNeuralPredictor(UINT64 PC, OpType opType, bool resolveDir, bool predDir, UINT64 branchTarget){

    #ifdef LOOPPREDICTOR
    // Update loop predictor and usefulness counter
    if (loopValid && (predDir != loopPred)) {
      WITHLOOP.update(loopPred == resolveDir);
    }
    ltable.update (PC, resolveDir, predDir != resolveDir, loopPred != predDir);
    #endif

    for(int j=GHL-1; j>0; j--)
    {
      SGHR[j] = SGHR[j-1];
      GHR[j] = GHR[j-1];
      HTrain[j] = HTrain[j-1];
      HA[j] = HA[j-1];
    }
    SGHR[0] = predDir;
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

        //First WL_1 branches: update T/NT weight tables separately
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
        //Next WL_2 branches: update regular weight tables
        for(int j = 0; j < WL_2; j++)	{
          uint64_t widx = gen_widx(PC, HA[j+WL_1+specul], WT_SIZE); // compute the index to access either weight table;
          if(t==GHR[specul+j+WL_1])
          { if(weight1[widx][j]<63) weight1[widx][j]++;}
          else
          { if(weight1[widx][j]>-64) weight1[widx][j]--;}
        }
        //Last WL_3 branches: update regular weight tables
        for(int j = 0; j < WL_3; j++)	{
          uint64_t widx = gen_widx(PC, HA[j+WL_1+WL_2+specul], WT_SIZE-1); // compute the index to access either weight table;
          if(t==GHR[specul+j+WL_1+WL_2])
          { if(weight2[widx][j]<63) weight2[widx][j]++;}
          else
          { if(weight2[widx][j]>-64) weight2[widx][j]--;}
        }
      }
      //Update threshold
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

      //////////////////////////////////////////////////
      // Branch history management.
      //////////////////////////////////////////////////

      // How many history bits are inserted for the buffer
      int maxt = 1;

      // Special treamtment for function call
      if (opType == OPTYPE_CALL_INDIRECT_COND) maxt = 3 ;
      if (opType == OPTYPE_CALL_DIRECT_COND) maxt = 5 ;

      // Branch history information
      // (This feature is derived from ISL-TAGE)
      int T = ((branchTarget ^ (branchTarget >> 3) ^ PC) << 1) + resolveDir;

      // Update global history and path history
      for (int t = 0; t < maxt; t++) {
        ghist.update((T >> t) & 1);
      }

      // Update local history
      lhist.update(PC, resolveDir);
    }

  }
  uint64_t gen_widx(uint64_t cur_pc, uint64_t path_pc, uint64_t wt_size){
    cur_pc = (cur_pc ) ^ (cur_pc / (1<<wt_size));
    path_pc = (path_pc) ^ (path_pc / (1<<wt_size));
    #ifdef GLOCAL
    uint64_t widx = cur_pc ^ (path_pc) ^ lhist.read(cur_pc, 24, wt_size);// ^ ghist.gidx(0, 128, wt_size);
    #else
    uint64_t widx = cur_pc ^ (path_pc);// ^ lhist.read(cur_pc, 24, wt_size) ^ ghist.gidx(0, 128, wt_size);
    #endif
    widx = widx % (1<<wt_size);
    return widx;
  }
  // Hash function that calculating index of weight tables
};

class PREDICTOR{

private:
  NeuralPredictor *neuralPred;

public:
  PREDICTOR(void);
  // The interface to the functions below CAN NOT be changed
  bool    GetPrediction(UINT64 PC, bool btbANSF, bool btbATSF, bool btbDYN);
  void    UpdatePredictor(UINT64 PC, OpType opType, bool resolveDir, bool predDir, UINT64 branchTarget, bool btbANSF, bool btbATSF, bool btbDYN);
  void    TrackOtherInst(UINT64 PC, OpType opType, bool branchDir, UINT64 branchTarget);

};



/***********************************************************/
#endif
