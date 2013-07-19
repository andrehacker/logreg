package de.tuberlin.dima.ml.pact;

import java.nio.ByteBuffer;
import java.nio.CharBuffer;
import java.nio.charset.CharacterCodingException;
import java.nio.charset.Charset;
import java.nio.charset.CharsetDecoder;
import java.util.Arrays;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import eu.stratosphere.nephele.configuration.Configuration;
import eu.stratosphere.pact.common.io.DelimitedInputFormat;
import eu.stratosphere.pact.common.io.TextInputFormat;
import eu.stratosphere.pact.common.type.PactRecord;
import eu.stratosphere.pact.common.type.base.PactString;

/**
 * Reads a file in the libsvm format (binary labels)
 * TODO This is not finished yet
 * Should read CharBuffer to String, split the string, and then convert to numbers
 * 
 */
public class LibsvmInputFormat extends DelimitedInputFormat {
    
    public static final String CHARSET_NAME = "textformat.charset";
    
    public static final String FIELD_POS = "textformat.pos";
    
    public static final String DEFAULT_CHARSET_NAME = "UTF-8";
    
    private static final Log LOG = LogFactory.getLog(TextInputFormat.class);
    
    protected final PactString theString = new PactString();
    
    protected CharsetDecoder decoder;
    
    protected ByteBuffer byteWrapper;
    
    protected boolean ascii;
    
    protected int pos;
    
    // --------------------------------------------------------------------------------------------
    
    @Override
    public void configure(Configuration parameters) {
        super.configure(parameters);
        
        // get the charset for the decoding
        String charsetName = parameters.getString(CHARSET_NAME, DEFAULT_CHARSET_NAME);
        if (charsetName == null || !Charset.isSupported(charsetName)) {
            throw new RuntimeException("Unsupported charset: " + charsetName);
        }
        
        // Latin-1 = 
        if (charsetName.equals("ISO-8859-1") || charsetName.equalsIgnoreCase("ASCII")) {
            this.ascii = true;
        } else {
            this.decoder = Charset.forName(charsetName).newDecoder();
            this.byteWrapper = ByteBuffer.allocate(1);
        }
        
        // get the field position to write in the record
        this.pos = parameters.getInteger(FIELD_POS, 0);
        if (this.pos < 0) {
            throw new RuntimeException("Illegal configuration value for the target position: " + this.pos);
        }
    }

    // --------------------------------------------------------------------------------------------

    public boolean readRecord(PactRecord target, byte[] bytes, int offset, int numBytes) {
        PactString str = this.theString;
        
        if (this.ascii) {
            str.setValueAscii(bytes, offset, numBytes);
        }
        else {
            ByteBuffer byteWrapper = this.byteWrapper;
            if (bytes != byteWrapper.array()) {
                byteWrapper = ByteBuffer.wrap(bytes, 0, bytes.length);
                this.byteWrapper = byteWrapper;
            }
            byteWrapper.position(offset);
            byteWrapper.limit(offset + numBytes);
                
            try {
                CharBuffer result = this.decoder.decode(byteWrapper);
                str.setValue(result);
            }
            catch (CharacterCodingException e) {
                byte[] copy = new byte[numBytes];
                System.arraycopy(bytes, offset, copy, 0, numBytes);
                LOG.warn("Line could not be encoded: " + Arrays.toString(copy), e);
                return false;
            }
        }
        
        target.clear();
        target.setField(this.pos, str);
        return true;
    }
}