package edu.berkeley.cs.nlp.ocular.model.emission;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import edu.berkeley.cs.nlp.ocular.data.textreader.Charset;
import edu.berkeley.cs.nlp.ocular.image.ImageUtils.PixelType;
import edu.berkeley.cs.nlp.ocular.model.CharacterTemplate;
import edu.berkeley.cs.nlp.ocular.model.em.EmissionCacheInnerLoop;
import edu.berkeley.cs.nlp.ocular.model.transition.SparseTransitionModel.TransitionState;
import tberg.murphy.gpu.CudaUtil;
import tberg.murphy.indexer.Indexer;
import tberg.murphy.threading.BetterThreader;

/**
 * @author Taylor Berg-Kirkpatrick (tberg@eecs.berkeley.edu)
 */
public class CachingEmissionModelExplicitOffset implements EmissionModel {
	
	private EmissionCacheInnerLoop innerLoop;
	private int numChars;
	private CharacterTemplate[] templates;
	private PixelType[][][] observations;
	private float[][] whiteObservations;
	private float[][] blackObservations;
	private int[][] templateAllowedWidths;
	private int[] templateMinWidths;
	private int[] templateMaxWidths;
	private int[] padAndTemplateMinWidths;
	private int[] padAndTemplateMaxWidths;
	private int[][] padAndTemplateAllowedWidths;
	private float[][][][][] cachedLogProbs;
	private int spaceIndex;
	private int padMinWidth;
	private int padMaxWidth;
	
	public CachingEmissionModelExplicitOffset(CharacterTemplate[] templates, Indexer<String> charIndexer, PixelType[][][] observations, int padMinWidth, int padMaxWidth, EmissionCacheInnerLoop innerLoop) {
		this.innerLoop = innerLoop;
		
		this.numChars = charIndexer.size();
		this.spaceIndex = charIndexer.getIndex(Charset.SPACE);
		this.templates = templates;
		this.observations = observations;
		this.padMinWidth = padMinWidth;
		this.padMaxWidth = padMaxWidth;
		
		this.whiteObservations = new float[observations.length][];
		this.blackObservations = new float[observations.length][];
		for (int d=0; d<observations.length; ++d) {
			this.whiteObservations[d] = new float[sequenceLength(d)*CharacterTemplate.LINE_HEIGHT];
			this.blackObservations[d] = new float[sequenceLength(d)*CharacterTemplate.LINE_HEIGHT];
			for (int t=0; t<sequenceLength(d); ++t) {
				for (int j=0; j<CharacterTemplate.LINE_HEIGHT; ++j) {
					PixelType observation = observations[d][t][j];
					if (observation == PixelType.BLACK) {
						this.whiteObservations[d][CudaUtil.flatten(sequenceLength(d), CharacterTemplate.LINE_HEIGHT, t, j)] = 0.0f;
						this.blackObservations[d][CudaUtil.flatten(sequenceLength(d), CharacterTemplate.LINE_HEIGHT, t, j)] = 1.0f;
					} else if (observation == PixelType.WHITE) {
						this.whiteObservations[d][CudaUtil.flatten(sequenceLength(d), CharacterTemplate.LINE_HEIGHT, t, j)] = 1.0f;
						this.blackObservations[d][CudaUtil.flatten(sequenceLength(d), CharacterTemplate.LINE_HEIGHT, t, j)] = 0.0f;
					} else {
						this.whiteObservations[d][CudaUtil.flatten(sequenceLength(d), CharacterTemplate.LINE_HEIGHT, t, j)] = 0.0f;
						this.blackObservations[d][CudaUtil.flatten(sequenceLength(d), CharacterTemplate.LINE_HEIGHT, t, j)] = 0.0f;
					} 
				}
			}
		}
	}
	
	public int numChars() {
		return numChars;
	}
	
	public int numSequences() {
		return observations.length;
	}
	
	public int sequenceLength(int d) {
		return observations[d].length;
	}
	
	public int[] allowedWidths(int c) {
		return padAndTemplateAllowedWidths[c];
	}
	
	public int[] allowedWidths(TransitionState ts) {
		return allowedWidths(ts.getGlyphChar().templateCharIndex);
	}
	
	
	public float logProb(int d, int t, int c, int w) {
		float result = Float.NEGATIVE_INFINITY;
		for (int offset=-CharacterTemplate.MAX_OFFSET; offset<=CharacterTemplate.MAX_OFFSET; ++offset) {
			result = Math.max(result, cachedLogProbs[d][t][c][offset+CharacterTemplate.MAX_OFFSET][w-padAndTemplateMinWidths[c]]);
		}
		return result;
	}
	
	public float logProb(int d, int t, TransitionState ts, int w) {
		int c = ts.getGlyphChar().templateCharIndex;
		int offset = ts.getOffset();
		return cachedLogProbs[d][t][c][offset+CharacterTemplate.MAX_OFFSET][w-padAndTemplateMinWidths[c]];
	}
	
	public int getExposure(int d, int t, TransitionState ts, int w) {
		int c = ts.getGlyphChar().templateCharIndex;
		int offset = ts.getOffset();
		double bestScore = Double.NEGATIVE_INFINITY;
		int bestExposure = -1;
		for (int e=0; e<CharacterTemplate.EXP_GAINS.length; ++e) {
			for (int pw=padMinWidth; pw<=padMaxWidth; ++pw) {
				int tw = w-pw;
				if (tw >= templateMinWidths[c] &&  tw <= templateMaxWidths[c]) {
					double score = templates[c].widthLogProb(tw) + templates[c].emissionLogProb(observations[d], t, t+tw, e, offset) + padWidthLogProb(pw) + templates[spaceIndex].emissionLogProb(observations[d], t+tw, t+tw+pw, e, offset);
					if (score > bestScore) {
						bestScore = score;
						bestExposure = e;
					}
				}
			}
		}
		return bestExposure;
	}
	
	public int getOffset(int d, int t, TransitionState ts, int w) {
		return ts.getOffset();
	}
	
	public int getPadWidth(int d, int t, TransitionState ts, int w) {
		int c = ts.getGlyphChar().templateCharIndex;
		int offset = ts.getOffset();
		double bestScore = Double.NEGATIVE_INFINITY;
		int bestPadWith = -1;
		for (int e=0; e<CharacterTemplate.EXP_GAINS.length; ++e) {
			for (int pw=padMinWidth; pw<=padMaxWidth; ++pw) {
				int tw = w-pw;
				if (tw >= templateMinWidths[c] &&  tw <= templateMaxWidths[c]) {
					double score = templates[c].widthLogProb(tw) + templates[c].emissionLogProb(observations[d], t, t+tw, e, offset) + padWidthLogProb(pw) + templates[spaceIndex].emissionLogProb(observations[d], t+tw, t+tw+pw, e, offset);
					if (score > bestScore) {
						bestScore = score;
						bestPadWith = pw;
					}
				}
			}
		}
		return bestPadWith;
	}
	
	public float padWidthLogProb(int pw) {
		return (float) Math.log(1.0 / ((padMaxWidth - padMinWidth) + 1.0)); 
	}
	
	public void rebuildCache() {
		long nanoTime = System.nanoTime();
		
		templateAllowedWidths = new int[numChars][];
		templateMinWidths = new int[numChars];
		templateMaxWidths = new int[numChars];
		padAndTemplateMinWidths = new int[numChars];
		padAndTemplateMaxWidths = new int[numChars];
		padAndTemplateAllowedWidths = new int[numChars][];
		for (int c=0; c<numChars; ++c) {
			templateAllowedWidths[c] = templates[c].allowedWidths();
			templateMinWidths[c] = templates[c].templateMinWidth();
			templateMaxWidths[c] = templates[c].templateMaxWidth();
			padAndTemplateMinWidths[c] = templates[c].templateMinWidth() + padMinWidth;
			padAndTemplateMaxWidths[c] = templates[c].templateMaxWidth() + padMaxWidth;
			boolean[] padAndTemplateAllowedWidthsBool = new boolean[padAndTemplateMaxWidths[c]+1];
			Arrays.fill(padAndTemplateAllowedWidthsBool, false);
			for (int tw : templateAllowedWidths[c]) {
				for (int pw=padMinWidth; pw<=padMaxWidth; ++pw) {
					padAndTemplateAllowedWidthsBool[tw+pw] = true;
				}
			}
			List<Integer> padAndTemplateAllowedWidthsList = new ArrayList<Integer>();
			for (int w=0; w<padAndTemplateAllowedWidthsBool.length; ++w) {
				if (padAndTemplateAllowedWidthsBool[w]) padAndTemplateAllowedWidthsList.add(w);
			}
			padAndTemplateAllowedWidths[c] = new int[padAndTemplateAllowedWidthsList.size()];
			for (int wi=0; wi<padAndTemplateAllowedWidthsList.size(); ++wi) {
				padAndTemplateAllowedWidths[c][wi] = padAndTemplateAllowedWidthsList.get(wi);
			}
		}

		final float[][][] logColumnProbsWhitespace = new float[observations.length][][];
		for (int d=0; d<observations.length; ++d) {
			logColumnProbsWhitespace[d] = new float[sequenceLength(d)][CharacterTemplate.EXP_GAINS.length];
			for (int e=0; e<CharacterTemplate.EXP_GAINS.length; ++e) {
				float[] logWhiteProbsWhitespace = templates[spaceIndex].logWhiteProbs(e, 0, 1)[0];
				float[] logBlackProbsWhitespace = templates[spaceIndex].logBlackProbs(e, 0, 1)[0];
				for (int t=0; t<sequenceLength(d); ++t) {
					float logProb = 0.0f;
					for (int j=0; j<CharacterTemplate.LINE_HEIGHT; ++j) {
						logProb += logWhiteProbsWhitespace[j] * whiteObservations[d][CudaUtil.flatten(sequenceLength(d), CharacterTemplate.LINE_HEIGHT, t, j)];
					}
					for (int j=0; j<CharacterTemplate.LINE_HEIGHT; ++j) {
						logProb += logBlackProbsWhitespace[j] * blackObservations[d][CudaUtil.flatten(sequenceLength(d), CharacterTemplate.LINE_HEIGHT, t, j)];
					}
					logColumnProbsWhitespace[d][t][e] = logProb;
				}
			}
		}
		
		cachedLogProbs = new float[numSequences()][][][][];
		for (int d=0; d<numSequences(); ++d) {
			cachedLogProbs[d] = new float[sequenceLength(d)][][][];
			for (int t=0; t<sequenceLength(d); ++t) {
				cachedLogProbs[d][t] = new float[numChars][][];
				for (int c=0; c<numChars; ++c) {
					cachedLogProbs[d][t][c] = new float[2*CharacterTemplate.MAX_OFFSET+1][];
					for (int offset=-CharacterTemplate.MAX_OFFSET; offset<=CharacterTemplate.MAX_OFFSET; ++offset) {
						cachedLogProbs[d][t][c][offset+CharacterTemplate.MAX_OFFSET] = new float[padAndTemplateMaxWidths[c]-padAndTemplateMinWidths[c]+1];
						Arrays.fill(cachedLogProbs[d][t][c][offset+CharacterTemplate.MAX_OFFSET], Float.NEGATIVE_INFINITY);
					}
				}
			}
		}
		
		int maxTemplateWidthTmp = Integer.MIN_VALUE;
		int minTemplateWidthTmp = Integer.MAX_VALUE;
		for (int c=0; c<numChars; ++c) maxTemplateWidthTmp = Math.max(maxTemplateWidthTmp, templateMaxWidths[c]);
		for (int c=0; c<numChars; ++c) minTemplateWidthTmp = Math.min(minTemplateWidthTmp, templateMinWidths[c]);
		final int maxTemplateWidth = maxTemplateWidthTmp;
		final int minTemplateWidth = minTemplateWidthTmp;
		final int numTemplateWidths = (maxTemplateWidth-minTemplateWidth)+1;
		final int[][][][] templateIndices = new int[numTemplateWidths][numChars][CharacterTemplate.EXP_GAINS.length][2*CharacterTemplate.MAX_OFFSET+1]; 
		@SuppressWarnings("unchecked")
		final List<float[]>[] whiteTemplatesList = new List[numTemplateWidths];
		@SuppressWarnings("unchecked")
		final List<float[]>[] blackTemplatesList = new List[numTemplateWidths];
		for (int tw=minTemplateWidth; tw<=maxTemplateWidth; ++tw) {
			whiteTemplatesList[tw-minTemplateWidth] = new ArrayList<float[]>();
			blackTemplatesList[tw-minTemplateWidth] = new ArrayList<float[]>();
		}
		final int[] templateNumIndices = new int[numTemplateWidths];
		for (int c=0; c<numChars; ++c) {
			for (int tw : templateAllowedWidths[c]) {
				for (int e=0; e<CharacterTemplate.EXP_GAINS.length; ++e) {
					for (int offset=-CharacterTemplate.MAX_OFFSET; offset<=CharacterTemplate.MAX_OFFSET; ++offset) {
						float[][] logWhiteProbsTemplate = templates[c].logWhiteProbs(e, offset, tw);
						float[][] logBlackProbsTemplate = templates[c].logBlackProbs(e, offset, tw);
						whiteTemplatesList[tw-minTemplateWidth].add(CudaUtil.flatten(logWhiteProbsTemplate));
						blackTemplatesList[tw-minTemplateWidth].add(CudaUtil.flatten(logBlackProbsTemplate));
						templateIndices[tw-minTemplateWidth][c][e][offset+CharacterTemplate.MAX_OFFSET] = templateNumIndices[tw-minTemplateWidth];
						templateNumIndices[tw-minTemplateWidth]++;
					}
				}
			}
		}
		int totalTemplateNumIndices = 0;
		final int[] templateIndicesOffsets = new int[numTemplateWidths];
		for (int tw=minTemplateWidth; tw<=maxTemplateWidth; ++tw) {
			templateIndicesOffsets[tw-minTemplateWidth] = totalTemplateNumIndices;
			totalTemplateNumIndices += templateNumIndices[tw-minTemplateWidth];
		}
		
		float[][] whiteTemplates = new float[numTemplateWidths][];
		float[][] blackTemplates = new float[numTemplateWidths][];
		for (int tw=minTemplateWidth; tw<=maxTemplateWidth; ++tw) {
			whiteTemplates[tw-minTemplateWidth] = CudaUtil.flatten(whiteTemplatesList[tw-minTemplateWidth]);
			blackTemplates[tw-minTemplateWidth] = CudaUtil.flatten(blackTemplatesList[tw-minTemplateWidth]);
		}
		
		int maxSequenceLength = Integer.MIN_VALUE;
		for (int d=0; d<numSequences(); ++d) maxSequenceLength = Math.max(maxSequenceLength, sequenceLength(d));
		
		innerLoop.startup(whiteTemplates, blackTemplates, templateNumIndices, templateIndicesOffsets, minTemplateWidth, maxTemplateWidth, maxSequenceLength, totalTemplateNumIndices);
		float[][] scores = new float[innerLoop.numOuterThreads()][maxSequenceLength*totalTemplateNumIndices];
		BetterThreader.Function<Integer,float[]> func = new BetterThreader.Function<Integer,float[]>(){public void call(Integer d, float[] scores){
			Arrays.fill(scores, 0.0f);
			innerLoop.compute(scores, whiteObservations[d], blackObservations[d], sequenceLength(d));
			populate(d, scores, minTemplateWidth, logColumnProbsWhitespace, templateIndices, templateNumIndices, templateIndicesOffsets, innerLoop.numPopulateThreads());
		}};
		BetterThreader<Integer,float[]> threader = new BetterThreader<Integer,float[]>(func, innerLoop.numOuterThreads());
		for (int d=0; d<numSequences(); ++d) threader.addFunctionArgument(d);
		for (int t=0; t<innerLoop.numOuterThreads(); ++t) threader.setThreadArgument(t, scores[t]);
		threader.run();
		innerLoop.shutdown();
		
		System.out.println("Rebuild emission cache: " + (System.nanoTime() - nanoTime)/1000000 + "ms");
		System.out.printf("Estimated emission cache size: %.3fgb\n", estimateMemoryUsage());
	}
	
	private void populate(final int d, final float[] scores, final int minTemplateWidth, final float[][][] logColumnProbsWhitespace, final int[][][][] templateIndices, final int[] templateNumIndices, final int[] templateIndicesOffsets, int numThreads) {
		BetterThreader.Function<Integer,Object> func = new BetterThreader.Function<Integer,Object>(){public void call(Integer t, Object ignore){
			for (int c=0; c<numChars; ++c) {
				int[] templateWidths = templateAllowedWidths[c];
				for (int tw : templateWidths) {
					double templateWidthLogProb = templates[c].widthLogProb(tw);
					if (t+tw+padMinWidth <= sequenceLength(d)) {
						for (int e=0; e<CharacterTemplate.EXP_GAINS.length; ++e) {
							float[] templateLogProbs = new float[CharacterTemplate.MAX_OFFSET*2+1];
							for (int offset=-CharacterTemplate.MAX_OFFSET; offset<=CharacterTemplate.MAX_OFFSET; ++offset) {
								templateLogProbs[offset+CharacterTemplate.MAX_OFFSET] = (float) templateWidthLogProb + scores[templateIndicesOffsets[tw-minTemplateWidth]*sequenceLength(d) + CudaUtil.flatten(sequenceLength(d), templateNumIndices[tw-minTemplateWidth], t, templateIndices[tw-minTemplateWidth][c][e][offset+CharacterTemplate.MAX_OFFSET])];
							}
							for (int pw=padMinWidth; pw<=padMaxWidth; ++pw) {
								int w = tw + pw;
								if (t+w <= sequenceLength(d)) {
									float padLogProb = (float) padWidthLogProb(pw);
									if (pw > 0) {
										for (int tt=0; tt<pw; ++tt) {
											padLogProb += logColumnProbsWhitespace[d][t+tw+tt][e];
										}
									}
									for (int offset=-CharacterTemplate.MAX_OFFSET; offset<=CharacterTemplate.MAX_OFFSET; ++offset) {
										float logProb = templateLogProbs[offset+CharacterTemplate.MAX_OFFSET] + padLogProb;
										if (logProb > cachedLogProbs[d][t][c][offset+CharacterTemplate.MAX_OFFSET][w-padAndTemplateMinWidths[c]]) {
											cachedLogProbs[d][t][c][offset+CharacterTemplate.MAX_OFFSET][w-padAndTemplateMinWidths[c]] = logProb;
										}
									}
								}
							}
						}
					}
				}
			}
		}};
		BetterThreader<Integer,Object> threader = new BetterThreader<Integer,Object>(func, numThreads);
		for (int t=0; t<sequenceLength(d); ++t) threader.addFunctionArgument(t);
		threader.run();
	}

	public void incrementCount(int d, TransitionState ts, int startCol, int endCol, float count) {
		if (count > 0.0) {
			int c = ts.getGlyphChar().templateCharIndex;
			int w = endCol - startCol;
			int tw = w - getPadWidth(d, startCol, ts, w);
			templates[c].incrementCounts(count, observations[d], startCol, tw, getExposure(d, startCol, ts, w), getOffset(d, startCol, ts, w));
		}
	}
	
	public void incrementCounts(int d, TransitionState[] ts, int[] widths) {
		int t=0;
		for (int i=0; i<ts.length; ++i) {
			int width = widths[i];
			incrementCount(d, ts[i], t, t+width, 1.0f);
			t += width;
		}
	}
	
	private double estimateMemoryUsage() {
		double elementsOfCache = 0.0;
		for (int i=0; i<cachedLogProbs.length; ++i) {
			if (cachedLogProbs[i] != null) {
				for (int j=0; j<cachedLogProbs[i].length; ++j) {
					if (cachedLogProbs[i][j] != null) {
						for (int k=0; k<cachedLogProbs[i][j].length; ++k) {
							if (cachedLogProbs[i][j][k] != null) {
								for (int l=0; l<cachedLogProbs[i][j][k].length; ++l) {
									if (cachedLogProbs[i][j][k][l] != null) elementsOfCache += cachedLogProbs[i][j][k][l].length;
								}
							}
						}
					}
				}
			}			
		}
		return 4 * elementsOfCache / 1e9;
	}
	
	public static class CachingEmissionModelExplicitOffsetFactory implements EmissionModel.EmissionModelFactory {
		Indexer<String> charIndexer;
		int padMinWidth;
		int padMaxWidth;
		EmissionCacheInnerLoop innerLoop;
		public CachingEmissionModelExplicitOffsetFactory(Indexer<String> charIndexer, int padMinWidth, int padMaxWidth, EmissionCacheInnerLoop innerLoop) {
			this.charIndexer = charIndexer;
			this.padMinWidth = padMinWidth;
			this.padMaxWidth = padMaxWidth;
			this.innerLoop = innerLoop;
		}
		public EmissionModel make(CharacterTemplate[] templates, PixelType[][][] observations) {
			return new CachingEmissionModelExplicitOffset(templates, charIndexer, observations, padMinWidth, padMaxWidth, innerLoop);
		}
	}
}
