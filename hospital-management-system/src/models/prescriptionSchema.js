/**
 * Extended Prescription Schema with Payment Integration
 * Extends existing prescription collection with payment tracking fields
 */

export const PrescriptionSchema = {
    // Existing fields (already implemented)
    prescription_id: String,
    doctor_id: String,
    patient_id: String,
    prescription_content: String,
    amount: Number, // From Document AI

    // New payment fields (Phase 1 extension)
    payment_status: {
        type: String,
        enum: ["pending", "paid", "failed"],
        default: "pending"
    },
    payment_method: {
        type: String,
        enum: ["cash", "upi", "card"],
        default: null
    },
    payment_timestamp: {
        type: Date,
        default: null
    },
    razorpay_payment_id: {
        type: String,
        default: null
    },
    processed_by_receptionist: {
        type: String,
        default: null
    },
    created_at: {
        type: Date,
        default: Date.now
    },
    updated_at: {
        type: Date,
        default: Date.now
    }
};

/**
 * New Payment Transactions Collection Schema
 * For detailed payment tracking and analytics
 */
export const PaymentTransactionSchema = {
    transaction_id: {
        type: String,
        required: true,
        unique: true
    },
    prescription_id: {
        type: String,
        required: true
    },
    patient_id: {
        type: String,
        required: true
    },
    amount: {
        type: Number,
        required: true
    },
    payment_method: {
        type: String,
        enum: ["cash", "upi", "card"],
        required: true
    },
    payment_status: {
        type: String,
        enum: ["pending", "paid", "failed"],
        default: "pending"
    },
    razorpay_payment_id: String,
    razorpay_order_id: String,
    timestamp: {
        type: Date,
        default: Date.now
    },
    processed_by: String, // receptionist_id for cash payments
    created_date: {
        type: Date,
        default: Date.now
    },
    updated_date: {
        type: Date,
        default: Date.now
    }
};
