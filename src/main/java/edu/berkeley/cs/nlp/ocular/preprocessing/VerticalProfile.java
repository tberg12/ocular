package edu.berkeley.cs.nlp.ocular.preprocessing;

import edu.berkeley.cs.nlp.ocular.preprocessing.VerticalModel.SuffStats;
import edu.berkeley.cs.nlp.ocular.preprocessing.VerticalModel.VerticalModelStateType;
import edu.berkeley.cs.nlp.ocular.image.ImageUtils;
import edu.berkeley.cs.nlp.ocular.image.ImageUtils.PixelType;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import tberg.murphy.math.m;
import tberg.murphy.tuple.Pair;

/**
 * @author Taylor Berg-Kirkpatrick (tberg@eecs.berkeley.edu)
 */
public class VerticalProfile {

	public static class VerticalSegmentation {
		public final int totalSize;
		// Contains pairs of state types and the indices at which they *start*
		public final List<Pair<VerticalModelStateType,Integer>> segments;

		public VerticalSegmentation(int totalSize, List<Pair<VerticalModelStateType,Integer>> segments) {
			this.totalSize = totalSize;
			this.segments = segments;
		}

		public VerticalModelStateType getType(int yIndex) {
			return segments.get(retrieveSegmentIndex(yIndex)).getFirst();
		}

		public int retrieveSegmentIndex(int yIndex) {
			for (int i = 0; i < segments.size(); i++) {
				if (segments.get(i).getSecond().intValue() > yIndex) {
					return i - 1;
				}
			}
			return segments.size() - 1;
		}

		public List<Pair<Integer,Integer>> retrieveLineBoundaries() {
			List<Pair<Integer,Integer>> lineBoundaries = new ArrayList<Pair<Integer,Integer>>();
			for (int i = 0; i < segments.size(); i++) {
				if (segments.get(i).getFirst() == VerticalModelStateType.BASE) {
					int startIdx = -1;
					int endIdx = -1;
					for (int j = i-1; j >= 0; j--) {
						// Change this to SPACE if you want to extract space above line
						if (segments.get(j).getFirst() == VerticalModelStateType.ASCENDER) {
							startIdx = segments.get(j).getSecond().intValue();
							break;
						}
					}
					if (startIdx == -1) {
						startIdx = 0;
					}
					// N.B. -1 because we have to advance one to get the end of that segment
					for (int j = i+1; j < segments.size() - 1; j++) {
						if (segments.get(j).getFirst() == VerticalModelStateType.DESCENDER) {
							endIdx = segments.get(j+1).getSecond().intValue();
							break;
						}
					}
					if (endIdx == -1) {
						endIdx = totalSize;
					}
					lineBoundaries.add(Pair.makePair(startIdx, endIdx));
				}
			}
			return lineBoundaries;
		}

		public List<Integer> retrieveBaselines() {
			List<Integer> baselines = new ArrayList<Integer>();
			for (int i = 0; i < segments.size(); i++) {
				if (segments.get(i).getFirst() == VerticalModelStateType.BASE) {
					if (i >= segments.size()-1) {
						baselines.add(totalSize);
					} else {
						baselines.add(segments.get(i+1).getSecond());
					}
				}
			}
			return baselines;
		}
	}

	public final double[][] image;
	public final double[] emissionsPerRow;

	public VerticalProfile(double[][] image) {
		this.image = image;
		this.emissionsPerRow = new double[image[0].length];
		for (int j = 0; j < image[0].length; j++) {
			double numBlackPixels = 0;
			for (int i = 0; i < image.length; i++) {
				if (ImageUtils.getPixelType(image[i][j]) == PixelType.BLACK) {
					numBlackPixels += 1;
				}
			}
			this.emissionsPerRow[j] = numBlackPixels;
		}
	}

	public static interface EMCallback {
		public void callback(VerticalModel model, VerticalProfile profile);
	}

	public VerticalModel runEM(int numItrs, int numRestarts) {
		return runEM(numItrs, numRestarts, null);
	}

	public VerticalModel runEM(int numItrs, int numRestarts, EMCallback callback) {
		double bestLogProb = Double.NEGATIVE_INFINITY;
		VerticalModel bestModel = null;
		Random rand = new Random(0);
		for (int r=0; r<numRestarts; ++r) {
			VerticalModel model = VerticalModel.getRandomlyInitializedModel(image.length, rand);
			double logNormalizer = Double.NEGATIVE_INFINITY;
			for (int itr = 0; itr < numItrs; itr++) {
//				System.out.println("ITERATION " + itr);
				double[][] alphas = computeAlphas(model, false);
				double[][] betas = computeBetas(model, false);
				logNormalizer = Double.NEGATIVE_INFINITY;
				for (int state = 0; state < model.numStates(); state++) {
					logNormalizer = m.logAdd(logNormalizer, alphas[alphas.length-1][state]);
				}
				SuffStats suffStats = new SuffStats();
				for (int state = 0; state < model.numStates(); state++) {
					for (int size = model.minSize(state); size < model.maxSize(state); size++) {
						for (int i = 0; i <= image[0].length - size; i++) {
							double logMassUnnorm = alphas[i][model.getPredecessor(state)] + model.getLogProb(this, state, i, i+size) + betas[i+size][state];
							double mass = Math.exp(logMassUnnorm - logNormalizer);
							suffStats.addIn(new SuffStats(this, state, i, i + size, mass));
						}
					}
				}
//				System.out.println("END OF ITERATION " + itr + ": NORMALIZER = " + logNormalizer);
				model.updateMeansOnly(suffStats.getEmissionMeans(), suffStats.getSizeMeans());
				if (callback != null) {
					callback.callback(model, this);
				}
			}
			if (logNormalizer > bestLogProb) {
				bestLogProb = logNormalizer;
				bestModel = model;
			}
		}
		return bestModel;
	}

	public VerticalSegmentation decode(VerticalModel model) {
		List<Pair<VerticalModelStateType,Integer>> segments = new ArrayList<Pair<VerticalModelStateType,Integer>>();
		double[][] alphas = computeAlphas(model, true);
		int currIdx = alphas.length - 1;
		int currState = -1;
		double currScore = Double.NEGATIVE_INFINITY;
		for (int state = 0; state < VerticalModelStateType.values().length; state++) {
			if (alphas[alphas.length - 1][state] > currScore) {
				currState = state;
				currScore = alphas[currIdx][state];
			}
		}
		while (currIdx > 0) {
			// Loop over lengths, since that's the only ambiguity
			int bestSize = -1;
			double bestSizeScore = Double.NEGATIVE_INFINITY;
			for (int size = model.minSize(currState); size < model.maxSize(currState); size++) {
				if (currIdx - size >= 0) {
					double score = alphas[currIdx - size][model.getPredecessor(currState)] +
							model.getLogProb(this, currState, currIdx - size, currIdx);
					if (score > bestSizeScore) {
						bestSize = size;
						bestSizeScore = score;
					}
				}
			}
			segments.add(0, Pair.makePair(VerticalModelStateType.values()[currState], currIdx - bestSize));
			currIdx = currIdx - bestSize;
			currState = model.getPredecessor(currState);
		}
		return new VerticalSegmentation(emissionsPerRow.length, segments);
	}

	// alphas[i][state] is the score of the best path with state *ending* at position i; means
	// the interpretation of the initial probs is a bit messed up but oh well
	private double[][] computeAlphas(VerticalModel model, boolean max) {
		int len = emissionsPerRow.length + 1;
		int numStates = model.numStates();
		double[][] alphas = new double[len][numStates];
		for (int i = 0; i < alphas.length; i++) {
			Arrays.fill(alphas[i], Double.NEGATIVE_INFINITY);
		}
		for (int j = 0; j < alphas[0].length; j++) {
			alphas[0][j] = Math.log(1.0/numStates);
		}
		for (int i = 0; i < emissionsPerRow.length; i++) {
			for (int state = 0; state < VerticalModelStateType.values().length; state++) {
				int prevState = model.getPredecessor(state);
				for (int size = model.minSize(state); size < model.maxSize(state); size++) {
					if (i + size <= emissionsPerRow.length) {
						double increment = alphas[i][prevState] + model.getLogProb(this, state, i, i + size);
						alphas[i+size][state] = sum(alphas[i+size][state], increment, max);
					}
				}
			}
		}
		return alphas;
	}

	private double[][] computeBetas(VerticalModel model, boolean max) {
		int len = emissionsPerRow.length + 1;
		int numStates = model.numStates();
		double[][] betas = new double[len][numStates];
		for (int i = 0; i < betas.length; i++) {
			Arrays.fill(betas[i], Double.NEGATIVE_INFINITY);
		}
		for (int j = 0; j < betas[emissionsPerRow.length].length; j++) {
			betas[emissionsPerRow.length][j] = 0.0;
		}
		for (int i = emissionsPerRow.length - 1; i >= 0; i--) {
			for (int state = 0; state < VerticalModelStateType.values().length; state++) {
				int nextState = model.getSuccessor(state);
				for (int size = model.minSize(nextState); size < model.maxSize(nextState); size++) {
					if (i + size <= emissionsPerRow.length) {
						double increment = model.getLogProb(this, nextState, i, i + size);
						betas[i][state] = sum(betas[i][state], betas[i+size][nextState] + increment, max);
					}
				}
			}
		}
		return betas;
	}

	private double sum(double a, double b, boolean max) {
		if (max) {
			return Math.max(a, b);
		} else {
			return m.logAdd(a, b);
		}
	}

}