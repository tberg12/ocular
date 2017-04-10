package edu.berkeley.cs.nlp.ocular.lm;

import java.util.Collection;

import edu.berkeley.cs.nlp.ocular.gsm.GlyphChar;
import edu.berkeley.cs.nlp.ocular.model.TransitionStateType;
import edu.berkeley.cs.nlp.ocular.model.transition.SparseTransitionModel;
import edu.berkeley.cs.nlp.ocular.model.transition.SparseTransitionModel.TransitionState;
import edu.berkeley.cs.nlp.ocular.util.Tuple2;

public class TempTransitionState implements SparseTransitionModel.TransitionState{
	private final int pos;
	private final TransitionStateType type;

	private final int lmCharIndex;
	
	public TempTransitionState(int pos, int character, TransitionStateType type) {
		this.pos = pos;
		this.type = type;
		
		this.lmCharIndex = character;		
	}

	@Override
	public int getLanguageIndex() {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public int getLmCharIndex() {
		// TODO Auto-generated method stub
		return this.lmCharIndex;
	}

	@Override
	public GlyphChar getGlyphChar() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public TransitionStateType getType() {
		// TODO Auto-generated method stub
		return this.type;
	}

	@Override
	public int getOffset() {
		// TODO Auto-generated method stub
		return this.pos;
	}

	@Override
	public int getExposure() {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public Collection<Tuple2<TransitionState, Double>> forwardTransitions() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public Collection<Tuple2<TransitionState, Double>> nextLineStartStates() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public double endLogProb() {
		// TODO Auto-generated method stub
		return 0;
	}
}
