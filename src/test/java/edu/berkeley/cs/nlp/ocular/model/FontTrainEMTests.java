package edu.berkeley.cs.nlp.ocular.model;

import static org.junit.Assert.assertEquals;

import java.util.Collection;
import java.util.List;

import org.junit.Test;

import edu.berkeley.cs.nlp.ocular.gsm.GlyphChar;
import edu.berkeley.cs.nlp.ocular.gsm.GlyphChar.GlyphType;
import edu.berkeley.cs.nlp.ocular.model.transition.SparseTransitionModel.TransitionState;
import edu.berkeley.cs.nlp.ocular.train.FontTrainer;
import edu.berkeley.cs.nlp.ocular.util.Tuple2;
import static edu.berkeley.cs.nlp.ocular.util.CollectionHelper.*;
import tberg.murphy.indexer.HashMapIndexer;
import tberg.murphy.indexer.Indexer;
import edu.berkeley.cs.nlp.ocular.model.DecodeState;
import static edu.berkeley.cs.nlp.ocular.model.TransitionStateType.*;

/**
 * @author Dan Garrette (dhgarrette@gmail.com)
 */
public class FontTrainEMTests {

	class TS implements TransitionState {
		public final int id;
		private int languageIndex;
		private int lmCharIndex;
		private TransitionStateType type;
		private GlyphChar glyphChar;
		
		public TS(int id, int languageIndex, int lmCharIndex, TransitionStateType type, GlyphChar glyphChar) {
			this.id = id;
			this.languageIndex = languageIndex;
			this.lmCharIndex = lmCharIndex;
			this.type = type;
			this.glyphChar = glyphChar;
		}
		@Override public int getLanguageIndex() { return languageIndex; }
		@Override public int getLmCharIndex() { return lmCharIndex; }
		@Override public TransitionStateType getType() { return type; }
		@Override public GlyphChar getGlyphChar() { return glyphChar; }
		
		@Override public int getOffset() { return -1; }
		@Override public int getExposure() { return -1; }
		@Override public Collection<Tuple2<TransitionState, Double>> forwardTransitions() { return null; }
		@Override public Collection<Tuple2<TransitionState, Double>> nextLineStartStates() { return null; }
		@Override public double endLogProb() { return -1; }
		
		@Override public String toString() {
			return "TS("+id+", "+languageIndex+", "+lmCharIndex+", "+type+", "+glyphChar+")";
		}
	}
	
	private DecodeState DS(TS ts) {
		return new DecodeState(ts, 0, 0, 0, 0);
	}
	
	@Test
	public void test_makeFullViterbiStateSeq() {

		Indexer<String> charIndexer = new HashMapIndexer<String>();
		charIndexer.index(new String[] { " ", "-", "a", "b", "c" });
		DecodeState[][] decodeStates = new DecodeState[][] {
			new DecodeState[]{	DS(new TS(1, -1, 0, LMRGN, new GlyphChar(0, GlyphType.NORMAL_CHAR))), 
								DS(new TS(2, -1, 0, LMRGN, new GlyphChar(0, GlyphType.NORMAL_CHAR))),
								DS(new TS(3, -1, 0, TMPL, new GlyphChar(0, GlyphType.NORMAL_CHAR))), 
								DS(new TS(4, 1, 2, TMPL, new GlyphChar(2, GlyphType.NORMAL_CHAR))),
								DS(new TS(5, 1, 3, TMPL, new GlyphChar(3, GlyphType.NORMAL_CHAR))), 
								DS(new TS(6, 1, 4, TMPL, new GlyphChar(4, GlyphType.NORMAL_CHAR))), 
								DS(new TS(7, 1, 1, RMRGN_HPHN_INIT, new GlyphChar(1, GlyphType.NORMAL_CHAR))), 
								DS(new TS(8, 1, 0, RMRGN_HPHN, new GlyphChar(0, GlyphType.NORMAL_CHAR))), 
								DS(new TS(9, 1, 0, RMRGN_HPHN, new GlyphChar(0, GlyphType.NORMAL_CHAR))) },
			new DecodeState[]{	DS(new TS(10, 1, 0, LMRGN_HPHN, new GlyphChar(0, GlyphType.NORMAL_CHAR))), 
								DS(new TS(11, 1, 0, LMRGN_HPHN, new GlyphChar(0, GlyphType.NORMAL_CHAR))),
								DS(new TS(12, 1, 0, TMPL, new GlyphChar(0, GlyphType.NORMAL_CHAR))),
								DS(new TS(13, 1, 2, TMPL, new GlyphChar(2, GlyphType.NORMAL_CHAR))),
								DS(new TS(14, 1, 3, TMPL, new GlyphChar(3, GlyphType.NORMAL_CHAR))),
								DS(new TS(15, 1, 4, TMPL, new GlyphChar(4, GlyphType.NORMAL_CHAR))),
								DS(new TS(16, 1, 0, RMRGN, new GlyphChar(0, GlyphType.NORMAL_CHAR))),
								DS(new TS(17, 1, 0, RMRGN, new GlyphChar(0, GlyphType.NORMAL_CHAR))) }
		};
		List<DecodeState> tsSeq = FontTrainer.makeFullViterbiStateSeq(decodeStates, charIndexer);
		List<Integer> expectedIds = makeList(2, 3, 4, 1);
		for (int i = 0; i < expectedIds.size(); ++i) {
			assertEquals(expectedIds.get(i).intValue(), ((TS)tsSeq.get(i).ts).id);
		}


	}
}
