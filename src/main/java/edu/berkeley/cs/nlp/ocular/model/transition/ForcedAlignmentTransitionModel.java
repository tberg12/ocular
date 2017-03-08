package edu.berkeley.cs.nlp.ocular.model.transition;

import static edu.berkeley.cs.nlp.ocular.util.Tuple2.Tuple2;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;

import com.sun.org.apache.bcel.internal.generic.LMUL;

import tberg.murphy.arrays.a;
import edu.berkeley.cs.nlp.ocular.data.textreader.CharIndexer;
import edu.berkeley.cs.nlp.ocular.data.textreader.Charset;
import edu.berkeley.cs.nlp.ocular.gsm.GlyphChar;
import edu.berkeley.cs.nlp.ocular.gsm.GlyphChar.GlyphType;
import edu.berkeley.cs.nlp.ocular.lm.FixedLanguageModel;
import edu.berkeley.cs.nlp.ocular.model.TransitionStateType;
import edu.berkeley.cs.nlp.ocular.model.transition.CharacterNgramTransitionModel.CharacterNgramTransitionState;
import edu.berkeley.cs.nlp.ocular.model.transition.SparseTransitionModel.TransitionState;
import edu.berkeley.cs.nlp.ocular.util.Tuple2;
import sun.security.util.Length;

/**
 * @author Taylor Berg-Kirkpatrick (tberg@eecs.berkeley.edu)
 */
public class ForcedAlignmentTransitionModel implements SparseTransitionModel {
	
	public class CharacterNgramTransitionState implements SparseTransitionModel.TransitionState {
		private final int context;
		private final TransitionStateType type;

		private final int lmCharIndex;
		
		public CharacterNgramTransitionState(int context, TransitionStateType type) {
			this.context = context;
			this.type = type;
			
			if (context == -1 || type == TransitionStateType.LMRGN || type == TransitionStateType.LMRGN_HPHN || type == TransitionStateType.RMRGN || type == TransitionStateType.RMRGN_HPHN) {
				this.lmCharIndex = spaceCharIndex;
			} else if (type == TransitionStateType.RMRGN_HPHN_INIT) {
				this.lmCharIndex = hyphenCharIndex;
			} else {
				this.lmCharIndex = lm.getCharAtPos(context);
			}	
		}
		
		public boolean equals(Object other) {
		    if (other instanceof CharacterNgramTransitionState) {
		    	CharacterNgramTransitionState that = (CharacterNgramTransitionState) other;
		    	if (this.type != that.type) {
		    		return false;
		    	} else if (this.context != that.context) {
		    		return false;
		    	} else {
		    		return true;
		    	}
		    } else {
		    	return false;
		    }
		}
		
		public int hashCode() {
			return 1013 * Integer.hashCode(context) + 1009 * this.type.ordinal();
		}
		
		public Collection<Tuple2<TransitionState,Double>> nextLineStartStates() {
			List<Tuple2<TransitionState,Double>> result = new ArrayList<Tuple2<TransitionState,Double>>();
			TransitionStateType type = getType();
			int context = getContext();
			int pos = context;
			int nextChar = lm.getCharAtPos(pos+1);
			
			if (type == TransitionStateType.TMPL) {
				
				if (nextChar == lm.getCharacterIndexer().getIndex(Charset.SPACE)) {
					{
						double score = Math.log(LINE_MRGN_PROB) + Math.log(lm.getCharNgramProb());
						if (score != Double.NEGATIVE_INFINITY) {
							result.add(Tuple2((TransitionState) new CharacterNgramTransitionState(pos+1, TransitionStateType.LMRGN), score));
						}
					}
					double score = Math.log((1.0 - LINE_MRGN_PROB)) + Math.log(lm.getCharNgramProb());
					if (score != Double.NEGATIVE_INFINITY) {
						result.add(Tuple2((TransitionState) new CharacterNgramTransitionState(pos+2, TransitionStateType.TMPL), score));
					}					
				}
			} else if (type == TransitionStateType.RMRGN) {
				{
					double score = Math.log(LINE_MRGN_PROB);
					if (score != Double.NEGATIVE_INFINITY) {
						result.add(Tuple2((TransitionState) new CharacterNgramTransitionState(pos, TransitionStateType.LMRGN), score));
					}
				}
				double score = Math.log((1.0 - LINE_MRGN_PROB)) + Math.log(lm.getCharNgramProb());
				if (score != Double.NEGATIVE_INFINITY) {
					result.add(Tuple2((TransitionState) new CharacterNgramTransitionState(pos+1, TransitionStateType.TMPL), score));
				}				
			} else if (type == TransitionStateType.RMRGN_HPHN || type == TransitionStateType.RMRGN_HPHN_INIT) {
				{
					double score = Math.log(LINE_MRGN_PROB);
					if (score != Double.NEGATIVE_INFINITY) {
						result.add(Tuple2((TransitionState) new CharacterNgramTransitionState(context, TransitionStateType.LMRGN_HPHN), score));
					}
				}
				double score = Math.log((1.0 - LINE_MRGN_PROB)) + Math.log(lm.getCharNgramProb());
				if (nextChar != spaceCharIndex && !isPunc[nextChar]) {
					result.add(Tuple2((TransitionState) new CharacterNgramTransitionState(pos+1, TransitionStateType.TMPL), score));
				}				
			} else if (type == TransitionStateType.LMRGN || type == TransitionStateType.LMRGN_HPHN) {
				{
					double score = Math.log(LINE_MRGN_PROB);
					if (score != Double.NEGATIVE_INFINITY) {
						result.add(Tuple2((TransitionState) new CharacterNgramTransitionState(pos, TransitionStateType.LMRGN), score));
					}
				}
				double score = Math.log((1.0 - LINE_MRGN_PROB)) + Math.log(lm.getCharNgramProb());
				if (score != Double.NEGATIVE_INFINITY) {
					result.add(Tuple2((TransitionState) new CharacterNgramTransitionState(pos+1, TransitionStateType.TMPL), score));
				}
			}
			return result;
		}
		
		public double endLogProb() {
			return 0.0;
		}
		
		public Collection<Tuple2<TransitionState,Double>> forwardTransitions() {
			int context = getContext();
			int pos = context;
			TransitionStateType type = getType();
			int nextChar = lm.getCharAtPos(pos+1);
			
			List<Tuple2<TransitionState,Double>> result = new ArrayList<Tuple2<TransitionState,Double>>();
			if (type == TransitionStateType.LMRGN) {
				{
					double score = Math.log(LINE_MRGN_PROB);
					if (score != Double.NEGATIVE_INFINITY) {
						result.add(Tuple2((TransitionState) new CharacterNgramTransitionState(context, TransitionStateType.LMRGN), score));
					}
				}
				result.add(Tuple2((TransitionState) new CharacterNgramTransitionState(pos+1, TransitionStateType.TMPL), Math.log((1.0 - LINE_MRGN_PROB)) + Math.log(lm.getCharNgramProb())));
			} else if (type == TransitionStateType.LMRGN_HPHN) {
				{
					double score = Math.log(LINE_MRGN_PROB);
					if (score != Double.NEGATIVE_INFINITY) {
						result.add(Tuple2((TransitionState) new CharacterNgramTransitionState(context, TransitionStateType.LMRGN_HPHN), score));
					}
				}
				if (nextChar != spaceCharIndex && !isPunc[nextChar]) {
					result.add(Tuple2((TransitionState) new CharacterNgramTransitionState(pos+1, TransitionStateType.TMPL), Math.log((1.0 - LINE_MRGN_PROB)) + Math.log(lm.getCharNgramProb())));					
				}
			} else if (type == TransitionStateType.RMRGN) {
				double score = Math.log(LINE_MRGN_PROB);
				if (score != Double.NEGATIVE_INFINITY) {
					result.add(Tuple2((TransitionState) new CharacterNgramTransitionState(pos, TransitionStateType.RMRGN), score));
				}
			} else if (type == TransitionStateType.RMRGN_HPHN) {
				double score = Math.log(LINE_MRGN_PROB);
				if (score != Double.NEGATIVE_INFINITY) {
					result.add(Tuple2((TransitionState) new CharacterNgramTransitionState(pos, TransitionStateType.RMRGN_HPHN), score));
				}
			} else if (type == TransitionStateType.RMRGN_HPHN_INIT) {
				double score = Math.log(LINE_MRGN_PROB);
				if (score != Double.NEGATIVE_INFINITY) {
					result.add(Tuple2((TransitionState) new CharacterNgramTransitionState(pos, TransitionStateType.RMRGN_HPHN), score));
				}
			} else if (type == TransitionStateType.TMPL) {	
				{
					double score = Math.log(LINE_MRGN_PROB) + Math.log(1.0 - LINE_END_HYPHEN_PROB) + Math.log(lm.getCharNgramProb());
					if (score != Double.NEGATIVE_INFINITY && nextChar == lm.getCharacterIndexer().getIndex(Charset.SPACE)) {
						result.add(Tuple2((TransitionState) new CharacterNgramTransitionState(pos+1, TransitionStateType.RMRGN), score));
					}
				}
				{
					double score = Math.log(LINE_MRGN_PROB) + Math.log(LINE_END_HYPHEN_PROB);
					if (score != Double.NEGATIVE_INFINITY && nextChar != lm.getCharacterIndexer().getIndex(Charset.SPACE)) {
						result.add(Tuple2((TransitionState) new CharacterNgramTransitionState(pos, TransitionStateType.RMRGN_HPHN_INIT), score));
					}
				}
				result.add(Tuple2((TransitionState) new CharacterNgramTransitionState(pos+1, TransitionStateType.TMPL), Math.log((1.0 - LINE_MRGN_PROB - SPACE_CONTINUE_PROB)) + Math.log(lm.getCharNgramProb())));	
				
				if (lm.getCharAtPos(pos) == lm.getCharacterIndexer().getIndex(Charset.SPACE)) {
					result.add(Tuple2((TransitionState) new CharacterNgramTransitionState(pos, TransitionStateType.TMPL), Math.log(SPACE_CONTINUE_PROB)));
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
		
		public int getContext() {
			return context;
		}
		
		public TransitionStateType getType() {
			return type;
		}
		
		public int getLanguageIndex() {
			return -1;
		}
	}
	
	public static final double LINE_MRGN_PROB = 0.3;
	public static final double LINE_END_HYPHEN_PROB = 1e-8;
	public static final double SPACE_CONTINUE_PROB = 1e-8;
	
	private FixedLanguageModel lm;
	private int spaceCharIndex;
	private int hyphenCharIndex;
	private boolean[] isPunc;

	public ForcedAlignmentTransitionModel(FixedLanguageModel lm) {
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
		int pos = 0;
		List<Tuple2<TransitionState,Double>> result = new ArrayList<Tuple2<TransitionState,Double>>();
		result.add(Tuple2((TransitionState) new CharacterNgramTransitionState(-1, TransitionStateType.LMRGN), Math.log(LINE_MRGN_PROB)));		
		result.add(Tuple2((TransitionState) new CharacterNgramTransitionState(pos, TransitionStateType.TMPL), Math.log((1.0 - LINE_MRGN_PROB)) + Math.log(lm.getCharNgramProb())));
		return result;
	}
	
	
}
