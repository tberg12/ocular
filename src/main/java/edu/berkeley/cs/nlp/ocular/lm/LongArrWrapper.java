package edu.berkeley.cs.nlp.ocular.lm;

import java.io.Serializable;
import java.util.Arrays;

/**
 * @author Taylor Berg-Kirkpatrick (tberg@eecs.berkeley.edu)
 */
public class LongArrWrapper implements Serializable {
	private static final long serialVersionUID = 5942433644698840887L;
	public final long[] arr;

	public LongArrWrapper(long[] arr) {
		this.arr = arr;
	}

	@Override
	public boolean equals(Object other) {
		if (other == null || !(other instanceof LongArrWrapper)) {
			return false;
		}
		return Arrays.equals(this.arr, ((LongArrWrapper)other).arr);
	}

	@Override
	public int hashCode() {
		return Arrays.hashCode(this.arr);
	}
}
