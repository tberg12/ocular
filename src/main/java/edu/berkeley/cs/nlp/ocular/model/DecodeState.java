package edu.berkeley.cs.nlp.ocular.model;

import edu.berkeley.cs.nlp.ocular.model.transition.SparseTransitionModel.TransitionState;

/**
 * @author Dan Garrette (dhgarrette@gmail.com)
 */
public class DecodeState {
	public final TransitionState ts;
	public final int charAndPadWidth;
	public final int charWidth;
	public final int padWidth;
	public final int exposure;
	public final int verticalOffset;
	
	public DecodeState(TransitionState ts, int charAndPadWidth, int padWidth, int exposure, int verticalOffset) {
		this.ts = ts;
		this.charAndPadWidth = charAndPadWidth;
		this.padWidth = padWidth;
		this.charWidth = charAndPadWidth - padWidth;
		this.exposure = exposure;
		this.verticalOffset = verticalOffset;
	}

}
