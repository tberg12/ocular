package edu.berkeley.cs.nlp.ocular.model;

import tberg.murphy.indexer.Indexer;
import tberg.murphy.indexer.IntArrayIndexer;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import tberg.murphy.math.m;
import tberg.murphy.opt.DifferentiableFunction;
import tberg.murphy.opt.LBFGSMinimizer;
import tberg.murphy.opt.Minimizer;
import tberg.murphy.tuple.Pair;
import tberg.murphy.arrays.a;
import edu.berkeley.cs.nlp.ocular.data.textreader.Charset;
import edu.berkeley.cs.nlp.ocular.image.ImageUtils.PixelType;
import edu.berkeley.cs.nlp.ocular.util.StringHelper;

/**
 * @author Taylor Berg-Kirkpatrick (tberg@eecs.berkeley.edu)
 */
public class CharacterTemplate implements Serializable {
	private static final long serialVersionUID = 2L;
	
	public static final int LINE_HEIGHT = 30;
	
	public static final float[] EXP_GAINS = new float[] {1.0f, 0.5f, 0.25f};
	public static final float[] EXP_STD_DEVS = new float[] {1.5f, 1.5f, 1.5f};
	public static final float[] EXP_SPC_BLACK_PROBS = new float[] {5e-2f, 2e-2f, 1e-1f};
	
	public static final int MAX_OFFSET = 5;
	
	public static final float EMIT_REG = 1e-2f;
	
	public static final float INIT_WIDTH_STD_THRESH = 2.5f;
	public static final float INIT_WIDTH_MIN_VAR = 1e-2f;
	public static final float LEARN_WIDTH_STD_THRESH = 2.5f;
	public static final float LEARN_WIDTH_MIN_VAR = 1e-2f;
	
	public static final float INIT_LBFGS_TOL = 1e-10f;
	public static final int INIT_LBFGS_ITERS = 1000;
	public static final float MSTEP_LBFGS_TOL = 1e-5f;
	public static final int MSTEP_LBFGS_ITERS = 20;
	
	private String character;
	
	private int templateMaxWidth;
	private int templateMinWidth;
	
	private float[][] templateWeights;
	private float[][] templateWeightsPriorMeans;

	private float[][][][] templateLogBlackProbs;
	private float[][][][] templateLogWhiteProbs;
	private boolean[][] templateCountSparsity;
	private boolean[][] templateLogProbsCached;
	
	private float[][][][] templateBlackCounts;
	private float[][][][] templateWhiteCounts;
	

	private float[] templateWidthProbs;

	private float[] templateWidthCounts;
	
	
	private Indexer<int[]> paramIndexer;

	private float[][][][] interpolationWeights;

	public float[][][][] getInterpolationWeights() {
		return interpolationWeights;
	}
	
	public CharacterTemplate(String character, float templateMaxWidthFraction, float templateMinWidthFraction) {
		this.templateMaxWidth = (int) Math.max(1, Math.floor(templateMaxWidthFraction*LINE_HEIGHT));
		this.templateMinWidth = (int) Math.max(1, Math.floor(templateMinWidthFraction*LINE_HEIGHT));
		
		int numTemplateWidths = (templateMaxWidth - templateMinWidth) + 1;
		this.templateWidthProbs = new float[numTemplateWidths];
		for (int i=0; i<templateWidthProbs.length; ++i) templateWidthProbs[i] = 1.0f;
		a.normalizei(templateWidthProbs);

		this.character = character;
		this.templateWidthCounts = new float[templateWidthProbs.length];
		
		if (!character.equals(Charset.SPACE)) {
			this.templateWeights = new float[templateMaxWidth][LINE_HEIGHT];
			for (int i=0; i<templateMaxWidth; ++i) {
				Arrays.fill(templateWeights[i], 0.0f);
			};

			this.templateWeightsPriorMeans = new float[templateMaxWidth][LINE_HEIGHT];
			for (int i=0; i<templateMaxWidth; ++i) {
				Arrays.fill(templateWeightsPriorMeans[i], 0.0f);
			};

			this.templateLogBlackProbs = new float[EXP_GAINS.length][templateWidthProbs.length][][];
			this.templateLogWhiteProbs = new float[EXP_GAINS.length][templateWidthProbs.length][][];
			this.templateLogProbsCached = new boolean[EXP_GAINS.length][templateWidthProbs.length];
			this.templateCountSparsity = new boolean[EXP_GAINS.length][templateWidthProbs.length];
			this.templateBlackCounts = new float[EXP_GAINS.length][templateWidthProbs.length][][];
			this.templateWhiteCounts = new float[EXP_GAINS.length][templateWidthProbs.length][][];
			this.interpolationWeights = new float[EXP_GAINS.length][templateWidthProbs.length][][];
			for (int e=0; e<EXP_GAINS.length; ++e) {
				for (int w=0; w<templateWidthProbs.length; ++w) {
					int width = templateMinWidth+w;
					this.interpolationWeights[e][w] = new float[width][templateMaxWidth];
					this.templateLogBlackProbs[e][w] = new float[width][LINE_HEIGHT];
					this.templateLogWhiteProbs[e][w] = new float[width][LINE_HEIGHT];
					this.templateBlackCounts[e][w] = new float[width][LINE_HEIGHT];
					this.templateWhiteCounts[e][w] = new float[width][LINE_HEIGHT];
					float interval = ((float) templateMaxWidth) / ((float) width);
					for (int i=0; i<width; ++i) {
						float emissionLocation = interval*(i+0.5f);
						for (int j=0; j<templateMaxWidth; ++j) {
							float templatePixelLocation = j+0.5f;
							this.interpolationWeights[e][w][i][j] = (float) Math.exp(m.gaussianLogProb((templatePixelLocation - emissionLocation)*(templatePixelLocation-emissionLocation), EXP_STD_DEVS[e]*interval));
						}
						a.normalizei(this.interpolationWeights[e][w][i]);
						a.scalei(this.interpolationWeights[e][w][i], EXP_GAINS[e]);
					}
				}
			}

			this.paramIndexer = new IntArrayIndexer();
			for (int i=0; i<this.templateWeights.length; ++i) {
				for (int j=0; j<this.templateWeights[i].length; ++j) {
					this.paramIndexer.getIndex(new int[] {i, j});
				}
			}
			this.paramIndexer.lock();
		}
	}
	
	public void initializeAndSetPriorFromFontData(PixelType[][][] fontData) {
		if (!character.equals(Charset.SPACE)) {
			System.out.println("Initializing "+character+" from font data...");
			clearEmissionCounts();
			clearWidthCounts();
			for (PixelType[][] observations : fontData) {
				if (observations.length >= templateMinWidth() && observations.length <= templateMaxWidth()) {
					incrementWidthCounts(observations.length, 1.0f);
					for (int pos=0; pos<observations.length; ++pos)
						incrementEmissionCounts(0, 0, observations.length, pos, 1.0f, observations[pos]);
				}
			}
			updateWidthParameters(INIT_WIDTH_MIN_VAR, INIT_WIDTH_STD_THRESH);
			updateEmissionParameters(INIT_LBFGS_TOL, INIT_LBFGS_ITERS);
			templateWeightsPriorMeans = a.copy(templateWeights);
			System.out.println(toString());
		}
	}
 	
	public int[] allowedWidths() {
		List<Integer> allowedWidths = new ArrayList<Integer>();
		for (int w=templateMinWidth(); w<=templateMaxWidth(); ++w) {
			if (widthProb(w) > 0.0f) {
				allowedWidths.add(w);
			}
		}
		return a.toIntArray(allowedWidths);
	}
	
	public float[][] blackProbs(int exposure, int offset, int width) {
		float[][] result = new float[width][LINE_HEIGHT];
		if (!character.equals(Charset.SPACE)) {
			for (int i=0; i<width; ++i) {
				for (int j=0; j<LINE_HEIGHT; ++j) {
					result[i][j] = (float) Math.exp(templateLogProbs(width, exposure, true)[i][Math.min(LINE_HEIGHT-1, Math.max(0, j+offset))]);
				}
			}
		} else {
			for (int i=0; i<width; ++i) {
				for (int j=0; j<LINE_HEIGHT; ++j) {
					result[i][j] = EXP_SPC_BLACK_PROBS[exposure];
				}
			}
		}
		return result;
	}
	
	public float[][] logBlackProbs(int exposure, int offset, int width) {
		float[][] result = new float[width][LINE_HEIGHT];
		if (!character.equals(Charset.SPACE)) {
			for (int i=0; i<width; ++i) {
				for (int j=0; j<LINE_HEIGHT; ++j) {
					result[i][j] = (float) templateLogProbs(width, exposure, true)[i][Math.min(LINE_HEIGHT-1, Math.max(0, j+offset))];
				}
			}
		} else {
			for (int i=0; i<width; ++i) {
				for (int j=0; j<LINE_HEIGHT; ++j) {
					result[i][j] = (float) Math.log(EXP_SPC_BLACK_PROBS[exposure]);
				}
			}
		}
		return result;
	}
	
	public float[][] logWhiteProbs(int exposure, int offset, int width) {
		float[][] result = new float[width][LINE_HEIGHT];
		if (!character.equals(Charset.SPACE)) {
			for (int i=0; i<width; ++i) {
				for (int j=0; j<LINE_HEIGHT; ++j) {
					result[i][j] = (float) templateLogProbs(width, exposure, false)[i][Math.min(LINE_HEIGHT-1, Math.max(0, j+offset))];
				}
			}
		} else {
			for (int i=0; i<width; ++i) {
				for (int j=0; j<LINE_HEIGHT; ++j) {
					result[i][j] = (float) Math.log(1.0 - EXP_SPC_BLACK_PROBS[exposure]);
				}
			}
		}
		return result;
	}
	
	public float emissionLogProb(PixelType[][] observations, int startCol, int endCol, int exposure, int offset) {
		int width = endCol - startCol;
		float logProb = 0.0f;
		for (int i=0; i<width; ++i) {
			logProb += columnEmissionLogProb(exposure, offset, width, i, observations[startCol+i]);
		}
		return logProb;
	}
	
	private float columnEmissionLogProb(int exposure, int offset, int width, int pos, PixelType[] observation) {
		float logProb = 0.0f;
		for (int j=0; j<LINE_HEIGHT; ++j) {
			logProb += pixelEmissionLogProb(exposure, offset, width, pos, j, observation[j]);
		}
		return logProb;
	}
	
	private float pixelEmissionLogProb(int exposure, int offset, int width, int pos, int j, PixelType observation) {
		if (!character.equals(Charset.SPACE)) {
			if (observation == PixelType.BLACK) {
				return templateLogProbs(width, exposure, true)[pos][Math.min(LINE_HEIGHT-1, Math.max(0, j+offset))];
			} if (observation == PixelType.WHITE) {
				return templateLogProbs(width, exposure, false)[pos][Math.min(LINE_HEIGHT-1, Math.max(0, j+offset))];
			} else {
				return 0.0f;
			} 
		} else {
			if (observation == PixelType.BLACK) {
				return (float) Math.log(EXP_SPC_BLACK_PROBS[exposure]);
			} if (observation == PixelType.WHITE) {
				return (float) Math.log(1.0 - EXP_SPC_BLACK_PROBS[exposure]);
			} else {
				return 0.0f;
			}
		}
	}
	
	public float widthProb(int width) {
		return templateWidthProbs[width-templateMinWidth()];
	}
	
	public float widthLogProb(int width) {
		return (float) Math.log(templateWidthProbs[width-templateMinWidth()]);
	}
	
	public void clearCounts() {
		clearEmissionCounts();
		clearWidthCounts();
	}
	
	public void incrementCounts(float count, PixelType[][] observations, int startCol, int width, int exposure, int offset) {
		for (int i=0; i<width; ++i) {
			incrementEmissionCounts(exposure, offset, width, i, count, observations[startCol+i]);
		}
		incrementWidthCounts(width, count);
	}
	
	public void updateParameters() {
		updateWidthParameters(LEARN_WIDTH_MIN_VAR, LEARN_WIDTH_STD_THRESH);
		updateEmissionParameters(MSTEP_LBFGS_TOL, MSTEP_LBFGS_ITERS);
	}

	public String getCharacter() {
		return character;
	}

	public String toString() {
		int bestWidth = -1;
		double bestWidthProb = Double.NEGATIVE_INFINITY;
		for (int width : allowedWidths()) {
			if (widthProb(width) > bestWidthProb) {
				bestWidthProb = widthProb(width);
				bestWidth = width;
			}
		}
		float[][] blackProbs = blackProbs(EXP_GAINS.length/2, 0, bestWidth);
		StringBuffer buf = new StringBuffer();
		buf.append(character).append("  ").append(StringHelper.toUnicode(character)).append(":\n");
		for (int j=0; j<LINE_HEIGHT; ++j) {
			for (int i=0; i<bestWidth; ++i) {
				float prob = blackProbs[i][j];
				if (prob >= 0.0 && prob < 0.333) {
					buf.append(". ");
				} else if (prob >= 0.333 && prob < 0.666) {
					buf.append("o ");
				} else if (prob >= 0.666) {
					buf.append("O ");
				}
			}
			buf.append("\n");
		}
		buf.append("Width probs: ").append(renderWidthProbs(templateWidthProbs, templateMinWidth())).append("\n");
		return buf.toString();
	}

	private String renderWidthProbs(float[] probs, int firstIndex) {
		if (probs.length <= 0) throw new RuntimeException("probs.length <= 0. was probs.length=" + probs.length);
		StringBuffer buf = new StringBuffer();
		for (int i=0; i<probs.length; ++i) {
			buf.append(i+firstIndex).append(" = ").append(String.format("%.2f", probs[i])).append(", ");
		}
		buf.delete(buf.length() - 2, buf.length());
		return buf.toString();
	}

	public int templateMaxWidth() {
		return templateMaxWidth;
	}

	public int templateMinWidth() {
		return templateMinWidth;
	}

	private void clearWidthCounts() {
		Arrays.fill(templateWidthCounts, 0.0f);
	}

	private void incrementWidthCounts(int width, float count) {
		synchronized (templateWidthCounts) {
			templateWidthCounts[width-templateMinWidth] += count;
		}
	}

	private void updateWidthParameters(float widthMinVar, float widthStdThresh) {
		if (!character.equals(Charset.SPACE)) {
			if (a.sum(templateWidthCounts) > 0.0) {
				float mean = 0.0f;
				float totalCount = a.sum(templateWidthCounts);
				for (int width=templateMinWidth; width<=templateMaxWidth; ++width) {
					mean += width * (templateWidthCounts[width-templateMinWidth] / totalCount);
				}
				float var = 0.0f;
				for (int width=templateMinWidth; width<=templateMaxWidth; ++width) {
					var += (mean - width) * (mean - width) * (templateWidthCounts[width-templateMinWidth] / totalCount);
				}
				templateWidthProbs = buildGuassianWidthProbs(mean, Math.max(widthMinVar, var), templateMinWidth, templateMaxWidth, widthStdThresh);
			}
		}
	}

	private static float[] buildGuassianWidthProbs(float mean, float var, int min, int max, float guassianWidthStdMultThreshold) {
		float[] probs = new float[max-min+1];
		for (int i=min; i<=max; ++i) {
			float sqrDistFromMean = (mean - i)*(mean - i);
			if (Math.sqrt(sqrDistFromMean) < guassianWidthStdMultThreshold*Math.sqrt(var)) {
				probs[i-min] = (float) Math.exp(-sqrDistFromMean/(2.0*var));
			}
		}
		a.normalizei(probs);		
		return probs;
	}

	private void clearEmissionCounts() {
		if (!character.equals(Charset.SPACE)) {
			for (int e=0; e<EXP_GAINS.length; ++e) {
				Arrays.fill(templateCountSparsity[e], false);
				for (int w=0; w<interpolationWeights[e].length; ++w) {
					for (int pos=0; pos<interpolationWeights[e][w].length; ++pos) {
						Arrays.fill(templateBlackCounts[e][w][pos], 0.0f);
						Arrays.fill(templateWhiteCounts[e][w][pos], 0.0f);
					}
				}
			}
		}
	}

	private void incrementEmissionCounts(int exposure, int offset, int width, int pos, float count, PixelType[] observation) {
		if (!character.equals(Charset.SPACE)) {
			synchronized (templateBlackCounts[exposure][width-templateMinWidth()][pos]) {
				for (int j=0; j<observation.length; ++j) {
					if (observation[j] == PixelType.BLACK) {
						templateBlackCounts[exposure][width-templateMinWidth()][pos][Math.min(LINE_HEIGHT-1, Math.max(0, j+offset))] += count;
					} else if (observation[j] == PixelType.WHITE) {
						templateWhiteCounts[exposure][width-templateMinWidth()][pos][Math.min(LINE_HEIGHT-1, Math.max(0, j+offset))] += count;
					}
				}
			}
			if (count > 0.0f) templateCountSparsity[exposure][width-templateMinWidth()] = true;
		}
	}
	
	private void updateEmissionParameters(float lbfgsTol, int iters) {
		if (!character.equals(Charset.SPACE)) {
			Minimizer minimizer = new LBFGSMinimizer(lbfgsTol, iters);
			double[] finalParams = minimizer.minimize(new NegExpectedLogLikelihoodFunc(), a.toDouble(getParamVector()), false, null);
			setParamVector(a.toFloat(finalParams));
		}
	}
	
	private void invalidateTemplateLogProbsCache() {
		for (int e=0; e<EXP_GAINS.length; ++e) {
			Arrays.fill(templateLogProbsCached[e], false);
		}
	}
	
	private float[][] templateLogProbs(int width, int e, boolean black) {
		if (!templateLogProbsCached[e][width-templateMinWidth()]) {
			for (int pos=0; pos<width; ++pos) {
				for (int j=0; j<LINE_HEIGHT; ++j) {
					float innerProd = 0.0f;
					for (int tpos=0; tpos<templateMaxWidth(); ++tpos) {
						innerProd += interpolationWeights[e][width-templateMinWidth()][pos][tpos]*templateWeights[tpos][j];
					}
					templateLogBlackProbs[e][width-templateMinWidth()][pos][j] = innerProd - (float) Math.log(1.0 + Math.exp(innerProd));
					templateLogWhiteProbs[e][width-templateMinWidth()][pos][j] = (float) -Math.log(1.0 + Math.exp(innerProd));
				}
			}
			templateLogProbsCached[e][width-templateMinWidth()] = true;
		}
		if (black) {
			return templateLogBlackProbs[e][width-templateMinWidth()];
		} else {
			return templateLogWhiteProbs[e][width-templateMinWidth()];
		}
	}

	private void setParamVector(float[] params) {
		for (int i=0; i<params.length; ++i) {
			int[]rowCol = paramIndexer.getObject(i);
			templateWeights[rowCol[0]][rowCol[1]] = params[i];
		}
		invalidateTemplateLogProbsCache();
	}

	private float[] getParamVector() {
		float[] params = new float[paramIndexer.size()];
		for (int i=0; i<templateWeights.length; ++i) {
			for (int j=0; j<templateWeights[i].length; ++j) {
				params[paramIndexer.getIndex(new int[] {i,j})] = templateWeights[i][j];
			}			
		}
		return params;
	}

	private float[] getPriorMeanVector() {
		float[] prior = new float[paramIndexer.size()];
		for (int i=0; i<templateWeightsPriorMeans.length; ++i) {
			for (int j=0; j<templateWeightsPriorMeans[i].length; ++j) {
				prior[paramIndexer.getIndex(new int[] {i,j})] = templateWeightsPriorMeans[i][j];
			}			
		}
		return prior;
	}

	private float getNegExpectedLogLikelihood() {
		float result = 0.0f;
		for (int e=0; e<EXP_GAINS.length; ++e) {
			for (int width=templateMinWidth(); width<=templateMaxWidth(); ++width) {
				if (templateCountSparsity[e][width-templateMinWidth()]) {
					for (int pos=0; pos<width; ++pos) {
						for (int j=0; j<LINE_HEIGHT; ++j) {
							result -= templateBlackCounts[e][width-templateMinWidth()][pos][j] * templateLogProbs(width, e, true)[pos][j] + templateWhiteCounts[e][width-templateMinWidth()][pos][j] * templateLogProbs(width, e, false)[pos][j];
						}
					}
				}
			}
		}
		return result;
	}

	private float[] getNegExpectedLogLikelihoodGradient() {
		float[] result = new float[paramIndexer.size()];
		for (int e=0; e<EXP_GAINS.length; ++e) {
			for (int width=templateMinWidth; width<=templateMaxWidth; ++width) {
				if (templateCountSparsity[e][width-templateMinWidth()]) {
					for (int pos=0; pos<width; ++pos) {
						for (int j=0; j<LINE_HEIGHT; ++j) {
							for (int tpos=0; tpos<templateMaxWidth; ++tpos) {
								int paramIndex = paramIndexer.getIndex(new int[] {tpos, j});
								result[paramIndex] -=  interpolationWeights[e][width-templateMinWidth()][pos][tpos] * (templateBlackCounts[e][width-templateMinWidth()][pos][j] - (templateBlackCounts[e][width-templateMinWidth()][pos][j] + templateWhiteCounts[e][width-templateMinWidth()][pos][j]) * Math.exp(templateLogProbs(width, e, true)[pos][j]));
							}
						}
					}
				}
			}
		}
		return result;
	}

	private class NegExpectedLogLikelihoodFunc implements DifferentiableFunction {
		float[] priorMeans = getPriorMeanVector();
		public Pair<Double, double[]> calculate(double[] xDouble) {
			float[] x = a.toFloat(xDouble);
			setParamVector(x);
			float[] priorDelta = a.comb(x, 1.0f, priorMeans, -1.0f);
			float reg = EMIT_REG*a.innerProd(priorDelta, priorDelta);
			float[] regGrad = a.scale(priorDelta, EMIT_REG*2.0f);
			return Pair.makePair((double) getNegExpectedLogLikelihood()+reg, a.toDouble(a.comb(getNegExpectedLogLikelihoodGradient(), 1.0f, regGrad, 1.0f)));
		}
	}

}
