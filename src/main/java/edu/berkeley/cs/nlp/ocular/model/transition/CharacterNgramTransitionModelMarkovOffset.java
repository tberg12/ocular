package edu.berkeley.cs.nlp.ocular.model.transition;

import static edu.berkeley.cs.nlp.ocular.util.Tuple2.Tuple2;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;

import tberg.murphy.arrays.a;
import edu.berkeley.cs.nlp.ocular.data.textreader.Charset;
import edu.berkeley.cs.nlp.ocular.gsm.GlyphChar;
import edu.berkeley.cs.nlp.ocular.gsm.GlyphChar.GlyphType;
import edu.berkeley.cs.nlp.ocular.lm.SingleLanguageModel;
import edu.berkeley.cs.nlp.ocular.model.CharacterTemplate;
import edu.berkeley.cs.nlp.ocular.model.TransitionStateType;
import edu.berkeley.cs.nlp.ocular.util.Tuple2;

/**
 * @author Taylor Berg-Kirkpatrick (tberg@eecs.berkeley.edu)
 */
public class CharacterNgramTransitionModelMarkovOffset implements SparseTransitionModel {
	
	public class CharacterNgramTransitionState implements SparseTransitionModel.TransitionState {
		private final int[] context;
		private final TransitionStateType type;
		private final int offset;

		private final int charIndex;
		private final int hashCode;
		
		public CharacterNgramTransitionState(int[] context, int offset, TransitionStateType type) {
			this.context = context;
			this.offset = offset;
			this.type = type;
			
			if (context.length == 0 || type == TransitionStateType.LMRGN || type == TransitionStateType.LMRGN_HPHN || type == TransitionStateType.RMRGN || type == TransitionStateType.RMRGN_HPHN) {
				this.charIndex = spaceCharIndex;
			} else if (type == TransitionStateType.RMRGN_HPHN_INIT) {
				this.charIndex = hyphenCharIndex;
			} else {
				this.charIndex = context[context.length-1];
			}
			
			this.hashCode = 1013 * Arrays.hashCode(context) + 1009 * this.offset + 997 * this.type.ordinal();
		}
		
		public boolean equals(Object other) {
		    if (other instanceof CharacterNgramTransitionState) {
		    	CharacterNgramTransitionState that = (CharacterNgramTransitionState) other;
		    	if (this.type != that.type) {
		    		return false;
		    	} else if (this.offset != that.offset) {
		    		return false;
		    	} else if (!Arrays.equals(this.context, that.context)) {
		    		return false;
		    	} else {
		    		return true;
		    	}
		    } else {
		    	return false;
		    }
		}
		
		public int hashCode() {
			return hashCode;
		}
		
		public Collection<Tuple2<TransitionState,Double>> nextLineStartStates() {
			List<Tuple2<TransitionState,Double>> result = new ArrayList<Tuple2<TransitionState,Double>>();
			TransitionStateType type = getType();
			int[] context = getContext();
			if (type == TransitionStateType.TMPL) {
				double scoreWithSpace =  Math.log(lm.getCharNgramProb(context, spaceCharIndex));
				if (scoreWithSpace != Double.NEGATIVE_INFINITY) {
					int[] contextWithSpace = shrinkContext(a.append(context, spaceCharIndex));
					{
						double score = Math.log(LINE_MRGN_PROB) + scoreWithSpace;
						if (score != Double.NEGATIVE_INFINITY) {
							result.add(Tuple2((TransitionState) new CharacterNgramTransitionState(contextWithSpace, 0, TransitionStateType.LMRGN), score));
						}
					}
					for (int c=0; c<lm.getCharacterIndexer().size(); ++c) {
						double intermediateScore = Math.log((1.0 - LINE_MRGN_PROB)) + scoreWithSpace + Math.log(lm.getCharNgramProb(contextWithSpace, c));
						if (intermediateScore != Double.NEGATIVE_INFINITY) {
							int[] nextContext = shrinkContext(a.append(contextWithSpace, c));
							for (int offset=-CharacterTemplate.MAX_OFFSET; offset<=CharacterTemplate.MAX_OFFSET; ++offset) {
								double score = intermediateScore + LOG_OFFSET_START_PROBS[offset+CharacterTemplate.MAX_OFFSET];
								if (score != Double.NEGATIVE_INFINITY) {
									result.add(Tuple2((TransitionState) new CharacterNgramTransitionState(nextContext, offset, TransitionStateType.TMPL), score));
								}
							}
						}
					}
				}
			} else if (type == TransitionStateType.RMRGN) {
				{
					double score = Math.log(LINE_MRGN_PROB);
					if (score != Double.NEGATIVE_INFINITY) {
						result.add(Tuple2((TransitionState) new CharacterNgramTransitionState(context, 0, TransitionStateType.LMRGN), score));
					}
				}
				for (int c=0; c<lm.getCharacterIndexer().size(); ++c) {
					double intermediateScore = Math.log((1.0 - LINE_MRGN_PROB)) + Math.log(lm.getCharNgramProb(context, c));
					if (intermediateScore != Double.NEGATIVE_INFINITY) {
						int[] nextContext = shrinkContext(a.append(context, c));
						for (int offset=-CharacterTemplate.MAX_OFFSET; offset<=CharacterTemplate.MAX_OFFSET; ++offset) {
							double score = intermediateScore + LOG_OFFSET_START_PROBS[offset+CharacterTemplate.MAX_OFFSET];
							if (score != Double.NEGATIVE_INFINITY) {
								result.add(Tuple2((TransitionState) new CharacterNgramTransitionState(nextContext, offset, TransitionStateType.TMPL), score));
							}
						}
					}
				}
			} else if (type == TransitionStateType.RMRGN_HPHN || type == TransitionStateType.RMRGN_HPHN_INIT) {
				{
					double score = Math.log(LINE_MRGN_PROB);
					if (score != Double.NEGATIVE_INFINITY) {
						result.add(Tuple2((TransitionState) new CharacterNgramTransitionState(context, 0, TransitionStateType.LMRGN_HPHN), score));
					}
				}
				for (int c=0; c<lm.getCharacterIndexer().size(); ++c) {
					if (c != spaceCharIndex && !isPunc[c]) {
						double intermedateScore = Math.log((1.0 - LINE_MRGN_PROB)) + Math.log(lm.getCharNgramProb(context, c));
						if (intermedateScore != Double.NEGATIVE_INFINITY) {
							int[] nextContext = shrinkContext(a.append(context, c));
							for (int offset=-CharacterTemplate.MAX_OFFSET; offset<=CharacterTemplate.MAX_OFFSET; ++offset) {
								double score = intermedateScore + LOG_OFFSET_START_PROBS[offset+CharacterTemplate.MAX_OFFSET];
								result.add(Tuple2((TransitionState) new CharacterNgramTransitionState(nextContext, offset, TransitionStateType.TMPL), score));
							}
						}
					}
				}
			} else if (type == TransitionStateType.LMRGN || type == TransitionStateType.LMRGN_HPHN) {
				{
					double score = Math.log(LINE_MRGN_PROB);
					if (score != Double.NEGATIVE_INFINITY) {
						result.add(Tuple2((TransitionState) new CharacterNgramTransitionState(new int[0], 0, TransitionStateType.LMRGN), score));
					}
				}
				for (int c=0; c<lm.getCharacterIndexer().size(); ++c) {
					double intermediateScore = Math.log((1.0 - LINE_MRGN_PROB)) + Math.log(lm.getCharNgramProb(context, c));
					if (intermediateScore != Double.NEGATIVE_INFINITY) {
						for (int offset=-CharacterTemplate.MAX_OFFSET; offset<=CharacterTemplate.MAX_OFFSET; ++offset) {
							double score = intermediateScore + LOG_OFFSET_START_PROBS[offset+CharacterTemplate.MAX_OFFSET];
							if (score != Double.NEGATIVE_INFINITY) {
								result.add(Tuple2((TransitionState) new CharacterNgramTransitionState(new int[] {c}, offset, TransitionStateType.TMPL), score));
							}
						}
					}
				}
			}
			return result;
		}
		
		public double endLogProb() {
			return 0.0;
		}
		
		public Collection<Tuple2<TransitionState,Double>> forwardTransitions() {
			int[] context = getContext();
			TransitionStateType type = getType();
			List<Tuple2<TransitionState,Double>> result = new ArrayList<Tuple2<TransitionState,Double>>();
			if (type == TransitionStateType.LMRGN) {
				{
					double score = Math.log(LINE_MRGN_PROB);
					if (score != Double.NEGATIVE_INFINITY) {
						result.add(Tuple2((TransitionState) new CharacterNgramTransitionState(context, 0, TransitionStateType.LMRGN), score));
					}
				}
				for (int c=0; c<lm.getCharacterIndexer().size(); ++c) {
					double intermediateScore = Math.log((1.0 - LINE_MRGN_PROB)) + Math.log(lm.getCharNgramProb(context, c));
					if (intermediateScore != Double.NEGATIVE_INFINITY) {
						int[] nextContext = shrinkContext(a.append(context, c));
						for (int offset=-CharacterTemplate.MAX_OFFSET; offset<=CharacterTemplate.MAX_OFFSET; ++offset) {
							double score = intermediateScore + LOG_OFFSET_START_PROBS[offset+CharacterTemplate.MAX_OFFSET];
							if (score != Double.NEGATIVE_INFINITY) {
								result.add(Tuple2((TransitionState) new CharacterNgramTransitionState(nextContext, offset, TransitionStateType.TMPL), score));
							}
						}
					}
				}
			} else if (type == TransitionStateType.LMRGN_HPHN) {
				{
					double score = Math.log(LINE_MRGN_PROB);
					if (score != Double.NEGATIVE_INFINITY) {
						result.add(Tuple2((TransitionState) new CharacterNgramTransitionState(context, 0, TransitionStateType.LMRGN_HPHN), score));
					}
				}
				for (int c=0; c<lm.getCharacterIndexer().size(); ++c) {
					if (c != spaceCharIndex && !isPunc[c]) {
						double intermediateScore = Math.log((1.0 - LINE_MRGN_PROB)) + Math.log(lm.getCharNgramProb(context, c));
						if (intermediateScore != Double.NEGATIVE_INFINITY) {
							int[] nextContext = shrinkContext(a.append(context, c));
							for (int offset=-CharacterTemplate.MAX_OFFSET; offset<=CharacterTemplate.MAX_OFFSET; ++offset) {
								double score = intermediateScore + LOG_OFFSET_START_PROBS[offset+CharacterTemplate.MAX_OFFSET];
								if (score != Double.NEGATIVE_INFINITY) {
									result.add(Tuple2((TransitionState) new CharacterNgramTransitionState(nextContext, offset, TransitionStateType.TMPL), score));
								}
							}
						}
					}
				}
			} else if (type == TransitionStateType.RMRGN) {
				double score = Math.log(LINE_MRGN_PROB);
				if (score != Double.NEGATIVE_INFINITY) {
					result.add(Tuple2((TransitionState) new CharacterNgramTransitionState(context, 0, TransitionStateType.RMRGN), score));
				}
			} else if (type == TransitionStateType.RMRGN_HPHN) {
				double score = Math.log(LINE_MRGN_PROB);
				if (score != Double.NEGATIVE_INFINITY) {
					result.add(Tuple2((TransitionState) new CharacterNgramTransitionState(context, 0, TransitionStateType.RMRGN_HPHN), score));
				}
			} else if (type == TransitionStateType.RMRGN_HPHN_INIT) {
				double score = Math.log(LINE_MRGN_PROB);
				if (score != Double.NEGATIVE_INFINITY) {
					result.add(Tuple2((TransitionState) new CharacterNgramTransitionState(context, 0, TransitionStateType.RMRGN_HPHN), score));
				}
			} else if (type == TransitionStateType.TMPL) {
				{
					double score = Math.log(LINE_MRGN_PROB) + Math.log(1.0 - LINE_END_HYPHEN_PROB) + Math.log(lm.getCharNgramProb(context, spaceCharIndex));
					if (score != Double.NEGATIVE_INFINITY) {
						int[] nextContext = shrinkContext(a.append(context, spaceCharIndex));
						result.add(Tuple2((TransitionState) new CharacterNgramTransitionState(nextContext, 0, TransitionStateType.RMRGN), score));
					}
				}
				double[] logOffsetTransProbs = LOG_OFFSET_TRANS_PROBS[getOffset()+CharacterTemplate.MAX_OFFSET];
				{
					double intermediateScore = Math.log(LINE_MRGN_PROB) + Math.log(LINE_END_HYPHEN_PROB);
					if (intermediateScore != Double.NEGATIVE_INFINITY) {
						for (int offset=Math.max(getOffset()-MAX_OFFSET_DIFF,-CharacterTemplate.MAX_OFFSET); offset<=Math.min(getOffset()+MAX_OFFSET_DIFF, CharacterTemplate.MAX_OFFSET); ++offset) {
							double score = intermediateScore + logOffsetTransProbs[offset+CharacterTemplate.MAX_OFFSET];
							if (score != Double.NEGATIVE_INFINITY) {
								result.add(Tuple2((TransitionState) new CharacterNgramTransitionState(context, offset, TransitionStateType.RMRGN_HPHN_INIT), score));
							}
						}
					}
				}
				for (int nextC=0; nextC<lm.getCharacterIndexer().size(); ++nextC) {
					double intermediateScore = Math.log((1.0 - LINE_MRGN_PROB)) + Math.log(lm.getCharNgramProb(context, nextC));
					if (intermediateScore != Double.NEGATIVE_INFINITY) {
						int[] nextContext = shrinkContext(a.append(context, nextC));
						for (int offset=Math.max(getOffset()-MAX_OFFSET_DIFF,-CharacterTemplate.MAX_OFFSET); offset<=Math.min(getOffset()+MAX_OFFSET_DIFF, CharacterTemplate.MAX_OFFSET); ++offset) {
							double score = intermediateScore + logOffsetTransProbs[offset+CharacterTemplate.MAX_OFFSET];
							if (score != Double.NEGATIVE_INFINITY) {
								result.add(Tuple2((TransitionState) new CharacterNgramTransitionState(nextContext, offset, TransitionStateType.TMPL), score));
							}
						}
					}
				}
			}
			return result;
		}
		
		public int getLmCharIndex() {
			return charIndex;
		}
		
		public GlyphChar getGlyphChar() {
			return new GlyphChar(charIndex, GlyphType.NORMAL_CHAR);
		}
		
		public int getOffset() {
			return offset;
		}
		
		public int getExposure() {
			throw new Error("Method not implemented");
		}
		
		public int[] getContext() {
			return context;
		}
		
		public TransitionStateType getType() {
			return type;
		}
		
		public int getLanguageIndex() {
			return -1;
		}
	}
	
	public static final double LINE_MRGN_PROB = 0.5;
	public static final double LINE_END_HYPHEN_PROB = 1e-8;
	public static int MAX_OFFSET_DIFF = 2;
	public static double MAX_OFFSET_TRANS_PROB_VAR = 0.05;
	public static final double[] LOG_OFFSET_START_PROBS = logOffsetStartProbs();
	public static final double[][] LOG_OFFSET_TRANS_PROBS = logOffsetTransProbs();
	
	private static double[] logOffsetStartProbs() {
		double[] offsetStartProbs = new double[CharacterTemplate.MAX_OFFSET*2+1];
		for (int offset0=-CharacterTemplate.MAX_OFFSET; offset0<=CharacterTemplate.MAX_OFFSET; ++offset0) {
			offsetStartProbs[offset0+CharacterTemplate.MAX_OFFSET] = 1.0;
		}
		a.logi(offsetStartProbs);
		return offsetStartProbs;
	}

	private static double[][] logOffsetTransProbs() {
		double[][] offsetTransProbs = new double[CharacterTemplate.MAX_OFFSET*2+1][CharacterTemplate.MAX_OFFSET*2+1];
		for (int offset0=-CharacterTemplate.MAX_OFFSET; offset0<=CharacterTemplate.MAX_OFFSET; ++offset0) {
			for (int offset1=-CharacterTemplate.MAX_OFFSET; offset1<=CharacterTemplate.MAX_OFFSET; ++offset1) {
				if (Math.abs(offset0 - offset1) <= MAX_OFFSET_DIFF) {
					double sqrDistFromMean = (offset0 - offset1)*(offset0 - offset1);
					offsetTransProbs[offset0+CharacterTemplate.MAX_OFFSET][offset1+CharacterTemplate.MAX_OFFSET] = Math.exp(-sqrDistFromMean/(2.0*MAX_OFFSET_TRANS_PROB_VAR));
				}
			}
		}
		a.normalizecoli(offsetTransProbs);
		a.logi(offsetTransProbs);
		return offsetTransProbs;
	}
	
	private SingleLanguageModel lm;
	private int spaceCharIndex;
	private int hyphenCharIndex;
	private boolean[] isPunc;

	public CharacterNgramTransitionModelMarkovOffset(SingleLanguageModel lm) {
		this.lm = lm;
		this.spaceCharIndex = lm.getCharacterIndexer().getIndex(Charset.SPACE);
		this.hyphenCharIndex = lm.getCharacterIndexer().getIndex(Charset.HYPHEN);
		this.isPunc = new boolean[lm.getCharacterIndexer().size()];
		Arrays.fill(this.isPunc, false);
		for (String c : lm.getCharacterIndexer().getObjects()) {
			if(Charset.isPunctuationChar(c))
				isPunc[lm.getCharacterIndexer().getIndex(c)] = true;
		}
	}
	
	public Collection<Tuple2<TransitionState,Double>> startStates() {
		List<Tuple2<TransitionState,Double>> result = new ArrayList<Tuple2<TransitionState,Double>>();
		result.add(Tuple2((TransitionState) new CharacterNgramTransitionState(new int[0], 0, TransitionStateType.LMRGN), Math.log(LINE_MRGN_PROB)));
		for (int c=0; c<lm.getCharacterIndexer().size(); ++c) {
			double intermediateScore = Math.log((1.0 - LINE_MRGN_PROB)) + Math.log(lm.getCharNgramProb(new int[0], c));
			if (intermediateScore != Double.NEGATIVE_INFINITY) {
				int[] nextContext = new int[] {c};
				for (int offset=-CharacterTemplate.MAX_OFFSET; offset<=CharacterTemplate.MAX_OFFSET; ++offset) {
					double score = intermediateScore + LOG_OFFSET_START_PROBS[offset+CharacterTemplate.MAX_OFFSET];
					if (score != Double.NEGATIVE_INFINITY) {
						result.add(Tuple2((TransitionState) new CharacterNgramTransitionState(nextContext, offset, TransitionStateType.TMPL), score));
					}
				}
			}
		}
		return result;
	}
	
	private int[] shrinkContext(int[] context) {
		return lm.shrinkContext(context);
	}
}
