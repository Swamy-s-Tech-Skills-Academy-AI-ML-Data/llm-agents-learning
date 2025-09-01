/**
 * Firebase Cloud Functions for Payment Module
 * Handles payment notifications and status updates
 */

const functions = require('firebase-functions');
const admin = require('firebase-admin');

// Initialize Firebase Admin if not already initialized
if (!admin.apps.length) {
    admin.initializeApp();
}

const db = admin.firestore();

/**
 * Phase 1: Payment Notification Function
 * Triggers when prescription is saved/updated with amount
 * Sends push notification to patient about payment due
 */
exports.sendPaymentNotification = functions.firestore
    .document('prescriptions/{prescriptionId}')
    .onWrite(async (change, context) => {
        try {
            const prescriptionId = context.params.prescriptionId;
            const newData = change.after.exists ? change.after.data() : null;
            const oldData = change.before.exists ? change.before.data() : null;

            // Only trigger notification for new prescriptions with amount or amount updates
            if (!newData || !newData.amount) {
                return null;
            }

            // Check if this is a new prescription or amount has changed
            const isNewPrescription = !oldData;
            const amountChanged = oldData && oldData.amount !== newData.amount;

            if (!isNewPrescription && !amountChanged) {
                return null;
            }

            // Get patient FCM token
            const patientDoc = await db.collection('patients').doc(newData.patient_id).get();
            if (!patientDoc.exists) {
                console.error('Patient not found:', newData.patient_id);
                return null;
            }

            const patientData = patientDoc.data();
            const fcmToken = patientData.fcm_token;

            if (!fcmToken) {
                console.error('FCM token not found for patient:', newData.patient_id);
                return null;
            }

            // Prepare notification payload
            const message = {
                token: fcmToken,
                notification: {
                    title: 'Payment Due',
                    body: `You have a payment of ₹${newData.amount} due for your prescription`
                },
                data: {
                    type: 'payment_due',
                    prescription_id: prescriptionId,
                    amount: newData.amount.toString(),
                    patient_id: newData.patient_id
                }
            };

            // Send notification
            const response = await admin.messaging().send(message);
            console.log('Payment notification sent successfully:', response);

            // Update prescription with notification sent timestamp
            await change.after.ref.update({
                notification_sent_at: admin.firestore.FieldValue.serverTimestamp(),
                updated_at: admin.firestore.FieldValue.serverTimestamp()
            });

            return response;
        } catch (error) {
            console.error('Error sending payment notification:', error);
            return null;
        }
    });

/**
 * Phase 1: Razorpay Webhook Handler
 * Processes payment confirmations and updates status
 */
exports.handleRazorpayWebhook = functions.https.onRequest(async (req, res) => {
    try {
        if (req.method !== 'POST') {
            return res.status(405).send('Method Not Allowed');
        }

        const crypto = require('crypto');
        const razorpayWebhookSecret = functions.config().razorpay.webhook_secret;

        // Verify webhook signature
        const signature = req.get('X-Razorpay-Signature');
        const body = JSON.stringify(req.body);
        const expectedSignature = crypto
            .createHmac('sha256', razorpayWebhookSecret)
            .update(body)
            .digest('hex');

        if (signature !== expectedSignature) {
            console.error('Invalid webhook signature');
            return res.status(400).send('Invalid signature');
        }

        const event = req.body.event;
        const paymentData = req.body.payload.payment.entity;

        if (event === 'payment.captured') {
            // Extract prescription_id from receipt
            const prescriptionId = paymentData.notes.prescription_id || paymentData.description;

            if (!prescriptionId) {
                console.error('Prescription ID not found in payment data');
                return res.status(400).send('Invalid payment data');
            }

            // Update prescription payment status
            await db.collection('prescriptions').doc(prescriptionId).update({
                payment_status: 'paid',
                payment_method: 'upi', // or 'card' based on payment method
                payment_timestamp: admin.firestore.FieldValue.serverTimestamp(),
                razorpay_payment_id: paymentData.id,
                updated_at: admin.firestore.FieldValue.serverTimestamp()
            });

            // Create payment transaction record
            const transactionId = `txn_${Date.now()}_${prescriptionId}`;
            await db.collection('payment_transactions').doc(transactionId).set({
                transaction_id: transactionId,
                prescription_id: prescriptionId,
                patient_id: paymentData.notes.patient_id,
                amount: paymentData.amount / 100, // Convert from paise to rupees
                payment_method: 'upi', // or determine from payment data
                payment_status: 'paid',
                razorpay_payment_id: paymentData.id,
                razorpay_order_id: paymentData.order_id,
                timestamp: admin.firestore.FieldValue.serverTimestamp(),
                created_date: admin.firestore.FieldValue.serverTimestamp(),
                updated_date: admin.firestore.FieldValue.serverTimestamp()
            });

            console.log('Payment captured successfully:', paymentData.id);
        } else if (event === 'payment.failed') {
            const prescriptionId = paymentData.notes.prescription_id || paymentData.description;

            if (prescriptionId) {
                await db.collection('prescriptions').doc(prescriptionId).update({
                    payment_status: 'failed',
                    updated_at: admin.firestore.FieldValue.serverTimestamp()
                });
            }
        }

        res.status(200).send('OK');
    } catch (error) {
        console.error('Error handling Razorpay webhook:', error);
        res.status(500).send('Internal Server Error');
    }
});

/**
 * Phase 2: Payment Status Updater for Cash Payments
 * Manual trigger from receptionist dashboard
 */
exports.updateCashPaymentStatus = functions.https.onCall(async (data, context) => {
    try {
        // Verify authentication
        if (!context.auth) {
            throw new functions.https.HttpsError('unauthenticated', 'User must be authenticated');
        }

        // Verify receptionist permissions
        const userDoc = await db.collection('users').doc(context.auth.uid).get();
        if (!userDoc.exists || userDoc.data().role !== 'receptionist') {
            throw new functions.https.HttpsError('permission-denied', 'User must be a receptionist');
        }

        const { prescriptionId, amount } = data;

        if (!prescriptionId || !amount) {
            throw new functions.https.HttpsError('invalid-argument', 'Prescription ID and amount are required');
        }

        // Update prescription payment status
        await db.collection('prescriptions').doc(prescriptionId).update({
            payment_status: 'paid',
            payment_method: 'cash',
            payment_timestamp: admin.firestore.FieldValue.serverTimestamp(),
            processed_by_receptionist: context.auth.uid,
            updated_at: admin.firestore.FieldValue.serverTimestamp()
        });

        // Create payment transaction record
        const transactionId = `txn_${Date.now()}_${prescriptionId}`;
        await db.collection('payment_transactions').doc(transactionId).set({
            transaction_id: transactionId,
            prescription_id: prescriptionId,
            patient_id: data.patient_id,
            amount: amount,
            payment_method: 'cash',
            payment_status: 'paid',
            timestamp: admin.firestore.FieldValue.serverTimestamp(),
            processed_by: context.auth.uid,
            created_date: admin.firestore.FieldValue.serverTimestamp(),
            updated_date: admin.firestore.FieldValue.serverTimestamp()
        });

        return { success: true, message: 'Payment status updated successfully' };
    } catch (error) {
        console.error('Error updating cash payment status:', error);
        throw new functions.https.HttpsError('internal', 'Failed to update payment status');
    }
});
