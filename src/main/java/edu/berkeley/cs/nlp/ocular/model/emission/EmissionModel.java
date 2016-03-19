package edu.berkeley.cs.nlp.ocular.model.emission;

import edu.berkeley.cs.nlp.ocular.image.ImageUtils.PixelType;
import edu.berkeley.cs.nlp.ocular.model.CharacterTemplate;
import edu.berkeley.cs.nlp.ocular.model.transition.SparseTransitionModel.TransitionState;

/**
 * @author Taylor Berg-Kirkpatrick (tberg@eecs.berkeley.edu)
 */
public interface EmissionModel {

	public abstract int numChars();

	public abstract int numSequences();

	public abstract int sequenceLength(int d);

	public abstract int[] allowedWidths(TransitionState ts);

	public abstract int[] allowedWidths(int c);

	public abstract float logProb(int d, int t, TransitionState ts, int w);

	public abstract float logProb(int d, int t, int c, int w);

	public abstract int getExposure(int d, int t, TransitionState ts, int w);

	public abstract int getOffset(int d, int t, TransitionState ts, int w);

	public abstract int getPadWidth(int d, int t, TransitionState ts, int w);

	public abstract float padWidthLogProb(int pw);

	public abstract void rebuildCache();

	public abstract void incrementCount(int d, TransitionState ts, int startCol, int endCol, float count);

	public abstract void incrementCounts(int d, TransitionState[] transitionStates, int[] widths);

	
	public static interface EmissionModelFactory {
		public EmissionModel make(CharacterTemplate[] templates, PixelType[][][] observations);
	}
	
}
