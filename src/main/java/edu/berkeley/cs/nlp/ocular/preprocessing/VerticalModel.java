package edu.berkeley.cs.nlp.ocular.preprocessing;

import java.util.Arrays;
import java.util.Random;

import tberg.murphy.arrays.a;
import tberg.murphy.math.m;

/**
 * @author Taylor Berg-Kirkpatrick (tberg@eecs.berkeley.edu)
 */
public class VerticalModel {

  public static enum VerticalModelStateType {
    ASCENDER, BASE, DESCENDER;
  }
  
  public static class SuffStats {
    public double[] totalMass = new double[VerticalModelStateType.values().length];
    public double[] totalMassTimesLength = new double[VerticalModelStateType.values().length];
    public double[] totalEmissionMass = new double[VerticalModelStateType.values().length];
    public double[] totalSizeMass = new double[VerticalModelStateType.values().length];
    
    public SuffStats() {
      Arrays.fill(totalMass, 0);
      Arrays.fill(totalMassTimesLength, 0);
      Arrays.fill(totalEmissionMass, 0);
      Arrays.fill(totalSizeMass, 0);
    }
    
    public SuffStats(VerticalProfile profile, int state, int start, int stop, double mass) {
      Arrays.fill(totalMass, 0);
      Arrays.fill(totalMassTimesLength, 0);
      Arrays.fill(totalEmissionMass, 0);
      Arrays.fill(totalSizeMass, 0);
      this.totalMass[state] += mass;
      this.totalMassTimesLength[state] += mass * (stop - start);
      for (int i = start; i < stop; i++) {
        this.totalEmissionMass[state] += mass * profile.emissionsPerRow[i];
      }
      this.totalSizeMass[state] += mass * (stop - start);
    }
    
    public void addIn(SuffStats other) {
      for (int i = 0; i < totalMass.length; i++) {
        totalMass[i] += other.totalMass[i];
      }
      for (int i = 0; i < totalMass.length; i++) {
        totalMassTimesLength[i] += other.totalMassTimesLength[i];
      }
      for (int i = 0; i < totalEmissionMass.length; i++) {
        totalEmissionMass[i] += other.totalEmissionMass[i];
      }
      for (int i = 0; i < totalSizeMass.length; i++) {
        totalSizeMass[i] += other.totalSizeMass[i];
      }
    }
    
    public double[] getEmissionMeans() {
      double[] emissionMeans = new double[totalSizeMass.length];
      for (int i = 0; i < emissionMeans.length; i++) {
        emissionMeans[i] = totalEmissionMass[i]/totalMassTimesLength[i];
      }
      return emissionMeans;
    }
    
    public double[] getSizeMeans() {
      double[] sizeMeans = new double[totalSizeMass.length];
      for (int i = 0; i < sizeMeans.length; i++) {
        sizeMeans[i] = totalSizeMass[i]/totalMass[i];
      }
      return sizeMeans;
    }
  }
  
  public int imageWidth;
  public double[][] emissionLogProbs;
  public double emissionVariance;
  public double[][] sizeLogProbs;
  public double[] sizeVariances;
  public static final int[] minSizes = { 6, 6, 6 };
  public static final int[] maxSizes = { 30, 30, 30 };
  
  public static VerticalModel getRandomlyInitializedModel(int imageWidth, Random rand) {
	  double[] emissionMeans = new double[VerticalModelStateType.values().length];
//	  emissionMeans[0] = 0.1 * imageWidth;
//	  emissionMeans[1] = 0.3 * imageWidth;
//	  emissionMeans[2] = 0.0 * imageWidth;
	  double[] blackFractions = new double[2];
	  for (int i = 0; i < blackFractions.length; i++) {
		  blackFractions[i] = 0.8 * rand.nextDouble();
	  }
	  Arrays.sort(blackFractions);
	  emissionMeans[0] = blackFractions[0] * imageWidth;
	  emissionMeans[1] = blackFractions[1] * imageWidth;
	  emissionMeans[2] = blackFractions[0] * imageWidth;
	  double emissionStd = 0.05;
	  double emissionVariance = (emissionStd * imageWidth) * (emissionStd * imageWidth);
	  double[] sizeMeans = new double[VerticalModelStateType.values().length];
	  double nonSpaceMean = rand.nextInt(Math.min(maxSizes[0], maxSizes[1])-Math.max(minSizes[0], minSizes[1])) + Math.max(minSizes[0], minSizes[1]);
	  double spaceMean = rand.nextInt(maxSizes[2]-minSizes[2]) + minSizes[2];
	  sizeMeans[0] = nonSpaceMean;
	  sizeMeans[1] = nonSpaceMean;
	  sizeMeans[2] = spaceMean;
	  double[] sizeVariances  = new double[VerticalModelStateType.values().length];
	  sizeVariances[0] = 2.0*2.0;
	  sizeVariances[1] = 2.0*2.0;
	  sizeVariances[2] = 2.0*2.0;
	  
	  return new VerticalModel(imageWidth, emissionMeans, emissionVariance, sizeMeans, sizeVariances);
  }
  
  public VerticalModel(int imageWidth, double[] emissionMeans, double emissionVariance, double[] sizeMeans, double[] sizeVariances) {
    this.imageWidth = imageWidth;
    this.emissionVariance = emissionVariance;
    this.sizeVariances = sizeVariances;
    updateMeansOnly(emissionMeans, sizeMeans);
  }
  
  public void updateMeansOnly(double[] emissionMeans, double[] sizeMeans) {
		  sizeVariances[0] = Math.pow(Math.sqrt(sizeVariances[0]) * 0.8, 2.0);
		  sizeVariances[1] = Math.pow(Math.sqrt(sizeVariances[1]) * 0.8, 2.0);
		  sizeVariances[2] = Math.pow(Math.sqrt(sizeVariances[2]) * 0.8, 2.0);
		  emissionVariance = Math.pow(Math.sqrt(emissionVariance) * 0.8, 2.0);
    setEmissionParams(emissionMeans, emissionVariance);
    setSizeParams(sizeMeans, sizeVariances);
//    System.out.println("Instantiating new model");
//    System.out.println("emissionMeans = " + Arrays.toString(emissionMeans));
//    System.out.println("emissionLogProbs[0] = " + Arrays.toString(emissionLogProbs[0]));
//    System.out.println("emissionLogProbs[1] = " + Arrays.toString(emissionLogProbs[1]));
//    System.out.println("emissionLogProbs[2] = " + Arrays.toString(emissionLogProbs[2]));
//    System.out.println("sizeMeans = " + Arrays.toString(sizeMeans));
//    System.out.println("sizeLogProbs[0] = " + Arrays.toString(sizeLogProbs[0]));
//    System.out.println("sizeLogProbs[1] = " + Arrays.toString(sizeLogProbs[1]));
//    System.out.println("sizeLogProbs[2] = " + Arrays.toString(sizeLogProbs[2]));
  }
  
  public void freezeSizeParams(int flexibilityRadius) {
    for (int i = 0; i < sizeLogProbs.length; i++) {
      int maxIdx = -1;
      double maxLogProb = Double.NEGATIVE_INFINITY;
      for (int j = 0; j < sizeLogProbs[i].length; j++) {
        if (sizeLogProbs[i][j] > maxLogProb) {
          maxIdx = j;
          maxLogProb = sizeLogProbs[i][j];
        }
      }
//      System.out.println("maxIdx: " + maxIdx);
      for (int j = 0; j < sizeLogProbs[i].length; j++) {
        if (j >= Math.max(0, maxIdx - flexibilityRadius) && j <= Math.min(sizeLogProbs[i].length, maxIdx + flexibilityRadius)) {
          sizeLogProbs[i][j] = 0.0;
        } else {
          sizeLogProbs[i][j] = Double.NEGATIVE_INFINITY;
        }
      }
      sizeLogProbs[i] = a.log(a.normalize(a.exp(sizeLogProbs[i])));
//      System.out.println(Arrays.toString(sizeLogProbs[i]));
      
    }
  }
  
  private void setEmissionParams(double[] emissionMeans, double emissionVariance) {
    this.emissionLogProbs = new double[emissionMeans.length][];
    for (int i = 0; i < emissionMeans.length; i++) {
      this.emissionLogProbs[i] = new double[imageWidth];
      for (int j = 0; j < this.emissionLogProbs[i].length; j++) {
        this.emissionLogProbs[i][j] = m.gaussianLogProb(emissionMeans[i], emissionVariance, j);
      }
      this.emissionLogProbs[i] = a.log(a.normalize(a.exp(this.emissionLogProbs[i])));
      // Smooth a little bit; only necessary if the variance isn't set high enough
//      final double SMOOTHING = 0.001;
//      this.emissionLogProbs[i] = a.log(a.add(a.scale(a.exp(this.emissionLogProbs[i]), 1.0 - SMOOTHING), SMOOTHING/this.emissionLogProbs[i].length));
//      assert Math.abs(a.sum(a.exp(this.emissionLogProbs[i])) - 1.0) < 1e-8 : a.sum(a.exp(this.emissionLogProbs[i]));
    }
  }
  
  private void setSizeParams(double[] sizeMeans, double[] sizeVariances) {
    this.sizeLogProbs = new double[sizeMeans.length][];
    for (int i = 0; i < sizeMeans.length; i++) {
      this.sizeLogProbs[i] = new double[maxSize(i) - minSize(i)];
      Arrays.fill(this.sizeLogProbs[i], 0.0);
      for (int j = 0; j < this.sizeLogProbs[i].length; j++) {
        this.sizeLogProbs[i][j] = m.gaussianLogProb(sizeMeans[i], sizeVariances[i], minSize(i) + j);
      }
      this.sizeLogProbs[i] = a.log(a.normalize(a.exp(this.sizeLogProbs[i])));
    }
  }
  
  public int numStates() {
    return VerticalModelStateType.values().length;
  }
  
  public int getPredecessor(int stateIdx) {
    return (stateIdx + numStates() - 1) % numStates();
  }
  
  public int getSuccessor(int stateIdx) {
    return (stateIdx + 1) % numStates();
  }
  
  public int minSize(int stateType) {
    return minSizes[stateType];
  }
  
  public int maxSize(int stateType) {
    return maxSizes[stateType];
  }
  
  public double getLogProb(VerticalProfile profile, int stateIdx, int posn) {
    int emissionIdx = (int)(Math.min(profile.emissionsPerRow[posn], emissionLogProbs[stateIdx].length - 1));
    return emissionLogProbs[stateIdx][emissionIdx];
  }
  
  public double getLogProb(VerticalProfile profile, int stateIdx, int start, int stop) {
//    System.out.println(start + " " + stop + " " + minSize(stateIdx));
    double totalLogProb = this.sizeLogProbs[stateIdx][stop-start-minSize(stateIdx)];
    for (int i = start; i < stop; i++) {
      totalLogProb += getLogProb(profile, stateIdx, i);
    }
    return totalLogProb;
  }
  
}
