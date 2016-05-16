package edu.berkeley.cs.nlp.ocular.model.em;

/**
 * @author Dan Garrette (dhgarrette@gmail.com)
 */
public class EmptyBeamException extends RuntimeException {
	private static final long serialVersionUID = 1L;

	public EmptyBeamException() {
		super();
	}

	public EmptyBeamException(String message, Throwable cause, boolean enableSuppression, boolean writableStackTrace) {
		super(message, cause, enableSuppression, writableStackTrace);
	}

	public EmptyBeamException(String message, Throwable cause) {
		super(message, cause);
	}

	public EmptyBeamException(String message) {
		super(message);
	}

	public EmptyBeamException(Throwable cause) {
		super(cause);
	}

}
