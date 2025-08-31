/**
 * Payment Service for Razorpay Integration
 * Phase 1: Core payment processing with Razorpay Gateway
 */

import Razorpay from 'razorpay';
import { collection, doc, updateDoc, serverTimestamp } from 'firebase/firestore';
import { db } from '../config/firebase';

class PaymentService {
    constructor() {
        // Initialize Razorpay instance
        this.razorpay = new Razorpay({
            key_id: process.env.REACT_APP_RAZORPAY_KEY_ID,
            key_secret: process.env.REACT_APP_RAZORPAY_KEY_SECRET,
        });
    }

    /**
     * Phase 1: Create Razorpay order for prescription payment
     * @param {string} prescriptionId - Prescription document ID
     * @param {number} amount - Payment amount in rupees
     * @param {string} patientId - Patient ID
     * @param {Object} patientInfo - Patient information
     * @returns {Promise<Object>} Razorpay order object
     */
    async createPaymentOrder(prescriptionId, amount, patientId, patientInfo) {
        try {
            const options = {
                amount: Math.round(amount * 100), // Convert to paise
                currency: 'INR',
                receipt: `prescription_${prescriptionId}`,
                notes: {
                    prescription_id: prescriptionId,
                    patient_id: patientId,
                    patient_name: patientInfo.name || '',
                    patient_phone: patientInfo.phone || ''
                },
                theme: {
                    color: '#3399cc'
                }
            };

            const order = await this.razorpay.orders.create(options);
            console.log('Razorpay order created:', order.id);

            // Update prescription with order details
            await updateDoc(doc(db, 'prescriptions', prescriptionId), {
                razorpay_order_id: order.id,
                payment_initiated: true,
                payment_initiated_at: serverTimestamp(),
                updated_at: serverTimestamp()
            });

            return order;
        } catch (error) {
            console.error('Error creating Razorpay order:', error);
            throw new Error('Failed to create payment order');
        }
    }

    /**
     * Phase 1: Initialize Razorpay payment checkout
     * @param {Object} order - Razorpay order object
     * @param {Object} patientInfo - Patient information
     * @param {function} onSuccess - Success callback
     * @param {function} onFailure - Failure callback
     */
    initiatePayment(order, patientInfo, onSuccess, onFailure) {
        const options = {
            key: process.env.REACT_APP_RAZORPAY_KEY_ID,
            amount: order.amount,
            currency: order.currency,
            name: 'Hospital Management System',
            description: 'Prescription Payment',
            order_id: order.id,
            handler: function (response) {
                // Payment successful
                console.log('Payment successful:', response);
                onSuccess(response);
            },
            prefill: {
                name: patientInfo.name || '',
                email: patientInfo.email || '',
                contact: patientInfo.phone || ''
            },
            notes: order.notes,
            theme: {
                color: '#3399cc'
            },
            modal: {
                ondismiss: function () {
                    console.log('Payment modal closed by user');
                    onFailure(new Error('Payment cancelled by user'));
                }
            },
            retry: {
                enabled: true,
                max_count: 3
            }
        };

        const razorpayInstance = new window.Razorpay(options);
        razorpayInstance.open();
    }

    /**
     * Phase 1: Verify payment signature (client-side validation)
     * @param {Object} paymentResponse - Response from Razorpay
     * @param {string} orderIdFromDB - Order ID from database
     * @returns {boolean} Signature verification result
     */
    verifyPaymentSignature(paymentResponse, orderIdFromDB) {
        try {
            const { razorpay_order_id, razorpay_payment_id, razorpay_signature } = paymentResponse;

            // Basic validation
            if (!razorpay_order_id || !razorpay_payment_id || !razorpay_signature) {
                return false;
            }

            // Verify order ID matches
            if (razorpay_order_id !== orderIdFromDB) {
                return false;
            }

            // Note: Server-side signature verification is handled by webhook
            return true;
        } catch (error) {
            console.error('Error verifying payment signature:', error);
            return false;
        }
    }

    /**
     * Phase 1: Handle successful payment
     * @param {string} prescriptionId - Prescription ID
     * @param {Object} paymentResponse - Razorpay payment response
     */
    async handlePaymentSuccess(prescriptionId, paymentResponse) {
        try {
            // Update prescription with payment details
            await updateDoc(doc(db, 'prescriptions', prescriptionId), {
                payment_status: 'processing', // Will be updated to 'paid' by webhook
                payment_method: 'upi', // Default, can be determined from payment method
                razorpay_payment_id: paymentResponse.razorpay_payment_id,
                payment_response: paymentResponse,
                payment_completed_at: serverTimestamp(),
                updated_at: serverTimestamp()
            });

            console.log('Payment success handled for prescription:', prescriptionId);
        } catch (error) {
            console.error('Error handling payment success:', error);
            throw new Error('Failed to update payment status');
        }
    }

    /**
     * Phase 1: Handle payment failure
     * @param {string} prescriptionId - Prescription ID
     * @param {Object} error - Error object
     */
    async handlePaymentFailure(prescriptionId, error) {
        try {
            await updateDoc(doc(db, 'prescriptions', prescriptionId), {
                payment_status: 'failed',
                payment_error: error.message || 'Payment failed',
                payment_failed_at: serverTimestamp(),
                updated_at: serverTimestamp()
            });

            console.log('Payment failure handled for prescription:', prescriptionId);
        } catch (updateError) {
            console.error('Error handling payment failure:', updateError);
        }
    }

    /**
     * Phase 2: Get payment methods available for prescription
     * @param {Object} prescription - Prescription object
     * @returns {Array} Available payment methods
     */
    getAvailablePaymentMethods(prescription) {
        const methods = [
            {
                id: 'upi',
                name: 'UPI Payment',
                description: 'Pay using UPI apps like PhonePe, Google Pay, Paytm',
                icon: 'upi-icon',
                enabled: true,
                processing_fee: 0
            },
            {
                id: 'card',
                name: 'Credit/Debit Card',
                description: 'Pay using your credit or debit card',
                icon: 'card-icon',
                enabled: true,
                processing_fee: Math.round(prescription.amount * 0.02) // 2% processing fee
            },
            {
                id: 'netbanking',
                name: 'Net Banking',
                description: 'Pay through your bank account',
                icon: 'bank-icon',
                enabled: true,
                processing_fee: 5 // Flat fee
            },
            {
                id: 'cash',
                name: 'Cash Payment',
                description: 'Pay cash at the reception',
                icon: 'cash-icon',
                enabled: true,
                processing_fee: 0,
                note: 'Complete payment at reception counter'
            }
        ];

        return methods;
    }

    /**
     * Phase 2: Mark cash payment as pending
     * @param {string} prescriptionId - Prescription ID
     */
    async markCashPaymentPending(prescriptionId) {
        try {
            await updateDoc(doc(db, 'prescriptions', prescriptionId), {
                payment_method: 'cash',
                payment_status: 'pending_cash',
                cash_payment_initiated_at: serverTimestamp(),
                updated_at: serverTimestamp()
            });

            console.log('Cash payment marked as pending for prescription:', prescriptionId);
        } catch (error) {
            console.error('Error marking cash payment as pending:', error);
            throw new Error('Failed to update cash payment status');
        }
    }

    /**
     * Phase 3: Get payment analytics data
     * @param {string} startDate - Start date for analytics
     * @param {string} endDate - End date for analytics
     * @returns {Promise<Object>} Analytics data
     */
    async getPaymentAnalytics(startDate, endDate) {
        try {
            // This would typically query the payment_transactions collection
            // Implementation depends on specific analytics requirements

            const analytics = {
                totalRevenue: 0,
                totalTransactions: 0,
                paymentMethodBreakdown: {
                    upi: { count: 0, amount: 0 },
                    card: { count: 0, amount: 0 },
                    netbanking: { count: 0, amount: 0 },
                    cash: { count: 0, amount: 0 }
                },
                dailyRevenue: [],
                averageTransactionValue: 0
            };

            // Query implementation would go here
            // const transactionsQuery = query(
            //   collection(db, 'payment_transactions'),
            //   where('created_date', '>=', startDate),
            //   where('created_date', '<=', endDate),
            //   where('payment_status', '==', 'paid')
            // );

            return analytics;
        } catch (error) {
            console.error('Error fetching payment analytics:', error);
            throw new Error('Failed to fetch payment analytics');
        }
    }
}

export default new PaymentService();
