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
import edu.berkeley.cs.nlp.ocular.model.TransitionStateType;
import edu.berkeley.cs.nlp.ocular.util.Tuple2;

/**
 * @author Taylor Berg-Kirkpatrick (tberg@eecs.berkeley.edu)
 */
public class CharacterNgramTransitionModel implements SparseTransitionModel {
	
	public class CharacterNgramTransitionState implements SparseTransitionModel.TransitionState {
		private final int[] context;
		private final TransitionStateType type;

		private final int lmCharIndex;
		
		public CharacterNgramTransitionState(int[] context, TransitionStateType type) {
			this.context = context;
			this.type = type;
			
			if (context.length == 0 || type == TransitionStateType.LMRGN || type == TransitionStateType.LMRGN_HPHN || type == TransitionStateType.RMRGN || type == TransitionStateType.RMRGN_HPHN) {
				this.lmCharIndex = spaceCharIndex;
			} else if (type == TransitionStateType.RMRGN_HPHN_INIT) {
				this.lmCharIndex = hyphenCharIndex;
			} else {
				this.lmCharIndex = context[context.length-1];
			}
		}
		
		public boolean equals(Object other) {
		    if (other instanceof CharacterNgramTransitionState) {
		    	CharacterNgramTransitionState that = (CharacterNgramTransitionState) other;
		    	if (this.type != that.type) {
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
			return 1013 * Arrays.hashCode(context) + 1009 * this.type.ordinal();
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
							result.add(Tuple2((TransitionState) new CharacterNgramTransitionState(contextWithSpace, TransitionStateType.LMRGN), score));
						}
					}
					for (int c=0; c<lm.getCharacterIndexer().size(); ++c) {
						double score = Math.log((1.0 - LINE_MRGN_PROB)) + scoreWithSpace + Math.log(lm.getCharNgramProb(contextWithSpace, c));
						if (score != Double.NEGATIVE_INFINITY) {
							int[] nextContext = shrinkContext(a.append(contextWithSpace, c));
							result.add(Tuple2((TransitionState) new CharacterNgramTransitionState(nextContext, TransitionStateType.TMPL), score));
						}
					}
				}
			} else if (type == TransitionStateType.RMRGN) {
				{
					double score = Math.log(LINE_MRGN_PROB);
					if (score != Double.NEGATIVE_INFINITY) {
						result.add(Tuple2((TransitionState) new CharacterNgramTransitionState(context, TransitionStateType.LMRGN), score));
					}
				}
				for (int c=0; c<lm.getCharacterIndexer().size(); ++c) {
					double score = Math.log((1.0 - LINE_MRGN_PROB)) + Math.log(lm.getCharNgramProb(context, c));
					if (score != Double.NEGATIVE_INFINITY) {
						int[] nextContext = shrinkContext(a.append(context, c));
						result.add(Tuple2((TransitionState) new CharacterNgramTransitionState(nextContext, TransitionStateType.TMPL), score));
					}
				}
			} else if (type == TransitionStateType.RMRGN_HPHN || type == TransitionStateType.RMRGN_HPHN_INIT) {
				{
					double score = Math.log(LINE_MRGN_PROB);
					if (score != Double.NEGATIVE_INFINITY) {
						result.add(Tuple2((TransitionState) new CharacterNgramTransitionState(context, TransitionStateType.LMRGN_HPHN), score));
					}
				}
				for (int c=0; c<lm.getCharacterIndexer().size(); ++c) {
					double score = Math.log((1.0 - LINE_MRGN_PROB)) + Math.log(lm.getCharNgramProb(context, c));
					if (c != spaceCharIndex && !isPunc[c]) {
						int[] nextContext = shrinkContext(a.append(context, c));
						result.add(Tuple2((TransitionState) new CharacterNgramTransitionState(nextContext, TransitionStateType.TMPL), score));
					}
				}
			} else if (type == TransitionStateType.LMRGN || type == TransitionStateType.LMRGN_HPHN) {
				{
					double score = Math.log(LINE_MRGN_PROB);
					if (score != Double.NEGATIVE_INFINITY) {
						result.add(Tuple2((TransitionState) new CharacterNgramTransitionState(new int[0], TransitionStateType.LMRGN), score));
					}
				}
				for (int c=0; c<lm.getCharacterIndexer().size(); ++c) {
					double score = Math.log((1.0 - LINE_MRGN_PROB)) + Math.log(lm.getCharNgramProb(context, c));
					if (score != Double.NEGATIVE_INFINITY) {
						result.add(Tuple2((TransitionState) new CharacterNgramTransitionState(new int[] {c}, TransitionStateType.TMPL), score));
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
						result.add(Tuple2((TransitionState) new CharacterNgramTransitionState(context, TransitionStateType.LMRGN), score));
					}
				}
				for (int c=0; c<lm.getCharacterIndexer().size(); ++c) {
					double score = Math.log((1.0 - LINE_MRGN_PROB)) + Math.log(lm.getCharNgramProb(context, c));
					if (score != Double.NEGATIVE_INFINITY) {
						int[] nextContext = shrinkContext(a.append(context, c));
						result.add(Tuple2((TransitionState) new CharacterNgramTransitionState(nextContext, TransitionStateType.TMPL), score));
					}
				}
			} else if (type == TransitionStateType.LMRGN_HPHN) {
				{
					double score = Math.log(LINE_MRGN_PROB);
					if (score != Double.NEGATIVE_INFINITY) {
						result.add(Tuple2((TransitionState) new CharacterNgramTransitionState(context, TransitionStateType.LMRGN_HPHN), score));
					}
				}
				for (int c=0; c<lm.getCharacterIndexer().size(); ++c) {
					double score = Math.log((1.0 - LINE_MRGN_PROB)) + Math.log(lm.getCharNgramProb(context, c));
					if (c != spaceCharIndex && !isPunc[c]) {
						int[] nextContext = shrinkContext(a.append(context, c));
						result.add(Tuple2((TransitionState) new CharacterNgramTransitionState(nextContext, TransitionStateType.TMPL), score));
					}
				}
			} else if (type == TransitionStateType.RMRGN) {
				double score = Math.log(LINE_MRGN_PROB);
				if (score != Double.NEGATIVE_INFINITY) {
					result.add(Tuple2((TransitionState) new CharacterNgramTransitionState(context, TransitionStateType.RMRGN), score));
				}
			} else if (type == TransitionStateType.RMRGN_HPHN) {
				double score = Math.log(LINE_MRGN_PROB);
				if (score != Double.NEGATIVE_INFINITY) {
					result.add(Tuple2((TransitionState) new CharacterNgramTransitionState(context, TransitionStateType.RMRGN_HPHN), score));
				}
			} else if (type == TransitionStateType.RMRGN_HPHN_INIT) {
				double score = Math.log(LINE_MRGN_PROB);
				if (score != Double.NEGATIVE_INFINITY) {
					result.add(Tuple2((TransitionState) new CharacterNgramTransitionState(context, TransitionStateType.RMRGN_HPHN), score));
				}
			} else if (type == TransitionStateType.TMPL) {
				{
					double score = Math.log(LINE_MRGN_PROB) + Math.log(1.0 - LINE_END_HYPHEN_PROB) + Math.log(lm.getCharNgramProb(context, spaceCharIndex));
					if (score != Double.NEGATIVE_INFINITY) {
						int[] nextContext = shrinkContext(a.append(context, spaceCharIndex));
						result.add(Tuple2((TransitionState) new CharacterNgramTransitionState(nextContext, TransitionStateType.RMRGN), score));
					}
				}
				{
					double score = Math.log(LINE_MRGN_PROB) + Math.log(LINE_END_HYPHEN_PROB);
					if (score != Double.NEGATIVE_INFINITY) {
						result.add(Tuple2((TransitionState) new CharacterNgramTransitionState(context, TransitionStateType.RMRGN_HPHN_INIT), score));
					}
				}
				for (int nextC=0; nextC<lm.getCharacterIndexer().size(); ++nextC) {
					double score = Math.log((1.0 - LINE_MRGN_PROB)) + Math.log(lm.getCharNgramProb(context, nextC));
					if (score != Double.NEGATIVE_INFINITY) {
						int[] nextContext = shrinkContext(a.append(context, nextC));
						result.add(Tuple2((TransitionState) new CharacterNgramTransitionState(nextContext, TransitionStateType.TMPL), score));
					}
				}
			}
			return result;
		}
		
		public int getLmCharIndex() {
			return lmCharIndex;
		}
		
		public GlyphChar getGlyphChar() {
			// Always render the character proposed by the language model
			return new GlyphChar(lmCharIndex, GlyphType.NORMAL_CHAR);
		}
		
		public int getOffset() {
			throw new Error("Method not implemented");
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
	
	private SingleLanguageModel lm;
	private int spaceCharIndex;
	private int hyphenCharIndex;
	private boolean[] isPunc;

	public CharacterNgramTransitionModel(SingleLanguageModel lm) {
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
		result.add(Tuple2((TransitionState) new CharacterNgramTransitionState(new int[0], TransitionStateType.LMRGN), Math.log(LINE_MRGN_PROB)));
		for (int c=0; c<lm.getCharacterIndexer().size(); ++c) {
			result.add(Tuple2((TransitionState) new CharacterNgramTransitionState(new int[] {c}, TransitionStateType.TMPL), Math.log((1.0 - LINE_MRGN_PROB)) + Math.log(lm.getCharNgramProb(new int[0], c))));
		}
		return result;
	}
	
	private int[] shrinkContext(int[] context) {
		return lm.shrinkContext(context);
	}
	
}
