/**
 * Patient Payment Component
 * Phase 1: Payment method selection and Razorpay integration
 */

import React, { useState, useEffect } from 'react';
import {
    View,
    Text,
    StyleSheet,
    TouchableOpacity,
    Alert,
    ActivityIndicator,
    ScrollView,
    Image
} from 'react-native';
import Icon from 'react-native-vector-icons/MaterialIcons';
import paymentService from '../../services/paymentService';

const PatientPaymentScreen = ({ route, navigation }) => {
    const { prescription, patientInfo } = route.params;
    const [selectedPaymentMethod, setSelectedPaymentMethod] = useState(null);
    const [paymentMethods, setPaymentMethods] = useState([]);
    const [loading, setLoading] = useState(false);
    const [processingPayment, setProcessingPayment] = useState(false);

    useEffect(() => {
        loadPaymentMethods();
    }, []);

    const loadPaymentMethods = () => {
        const methods = paymentService.getAvailablePaymentMethods(prescription);
        setPaymentMethods(methods);
    };

    const handlePaymentMethodSelect = (method) => {
        setSelectedPaymentMethod(method);
    };

    const calculateTotalAmount = () => {
        if (!selectedPaymentMethod) return prescription.amount;

        return prescription.amount + (selectedPaymentMethod.processing_fee || 0);
    };

    const handlePaymentInitiation = async () => {
        if (!selectedPaymentMethod) {
            Alert.alert('Error', 'Please select a payment method');
            return;
        }

        if (selectedPaymentMethod.id === 'cash') {
            handleCashPayment();
            return;
        }

        await processOnlinePayment();
    };

    const handleCashPayment = async () => {
        try {
            setLoading(true);
            await paymentService.markCashPaymentPending(prescription.id);

            Alert.alert(
                'Cash Payment Selected',
                'Please visit the reception counter to complete your payment. Your prescription will be ready after payment confirmation.',
                [
                    {
                        text: 'OK',
                        onPress: () => navigation.goBack()
                    }
                ]
            );
        } catch (error) {
            Alert.alert('Error', 'Failed to process cash payment selection');
        } finally {
            setLoading(false);
        }
    };

    const processOnlinePayment = async () => {
        try {
            setProcessingPayment(true);

            // Create Razorpay order
            const order = await paymentService.createPaymentOrder(
                prescription.id,
                calculateTotalAmount(),
                patientInfo.id,
                patientInfo
            );

            // Initialize Razorpay payment
            paymentService.initiatePayment(
                order,
                patientInfo,
                handlePaymentSuccess,
                handlePaymentFailure
            );
        } catch (error) {
            console.error('Payment initiation error:', error);
            Alert.alert('Error', 'Failed to initiate payment. Please try again.');
        } finally {
            setProcessingPayment(false);
        }
    };

    const handlePaymentSuccess = async (paymentResponse) => {
        try {
            setLoading(true);

            // Verify payment signature
            const isValid = paymentService.verifyPaymentSignature(
                paymentResponse,
                paymentResponse.razorpay_order_id
            );

            if (!isValid) {
                throw new Error('Payment verification failed');
            }

            // Handle successful payment
            await paymentService.handlePaymentSuccess(prescription.id, paymentResponse);

            Alert.alert(
                'Payment Successful',
                'Your payment has been processed successfully. Your prescription is now ready for pickup.',
                [
                    {
                        text: 'OK',
                        onPress: () => navigation.goBack()
                    }
                ]
            );
        } catch (error) {
            console.error('Payment success handling error:', error);
            Alert.alert('Error', 'Payment completed but failed to update records. Please contact support.');
        } finally {
            setLoading(false);
        }
    };

    const handlePaymentFailure = async (error) => {
        try {
            await paymentService.handlePaymentFailure(prescription.id, error);
            Alert.alert('Payment Failed', error.message || 'Payment was cancelled or failed. Please try again.');
        } catch (updateError) {
            console.error('Payment failure handling error:', updateError);
            Alert.alert('Error', 'Payment failed and could not update records. Please contact support.');
        }
    };

    const PaymentMethodCard = ({ method, isSelected, onSelect }) => (
        <TouchableOpacity
            style={[styles.paymentMethodCard, isSelected && styles.selectedPaymentMethod]}
            onPress={() => onSelect(method)}
            disabled={!method.enabled}
        >
            <View style={styles.paymentMethodHeader}>
                <Icon name={getPaymentMethodIcon(method.id)} size={24} color="#3399cc" />
                <View style={styles.paymentMethodInfo}>
                    <Text style={styles.paymentMethodName}>{method.name}</Text>
                    <Text style={styles.paymentMethodDescription}>{method.description}</Text>
                    {method.note && (
                        <Text style={styles.paymentMethodNote}>{method.note}</Text>
                    )}
                </View>
            </View>

            {method.processing_fee > 0 && (
                <View style={styles.processingFeeContainer}>
                    <Text style={styles.processingFeeText}>
                        Processing Fee: ₹{method.processing_fee}
                    </Text>
                </View>
            )}

            {isSelected && (
                <View style={styles.selectedIndicator}>
                    <Icon name="check-circle" size={20} color="#4CAF50" />
                </View>
            )}
        </TouchableOpacity>
    );

    const getPaymentMethodIcon = (methodId) => {
        const iconMap = {
            upi: 'smartphone',
            card: 'credit-card',
            netbanking: 'account-balance',
            cash: 'money'
        };
        return iconMap[methodId] || 'payment';
    };

    if (loading || processingPayment) {
        return (
            <View style={styles.loadingContainer}>
                <ActivityIndicator size="large" color="#3399cc" />
                <Text style={styles.loadingText}>
                    {processingPayment ? 'Processing Payment...' : 'Loading...'}
                </Text>
            </View>
        );
    }

    return (
        <ScrollView style={styles.container}>
            {/* Prescription Summary */}
            <View style={styles.prescriptionSummary}>
                <Text style={styles.sectionTitle}>Prescription Payment</Text>
                <View style={styles.summaryRow}>
                    <Text style={styles.summaryLabel}>Prescription ID:</Text>
                    <Text style={styles.summaryValue}>{prescription.id}</Text>
                </View>
                <View style={styles.summaryRow}>
                    <Text style={styles.summaryLabel}>Doctor:</Text>
                    <Text style={styles.summaryValue}>{prescription.doctor_name}</Text>
                </View>
                <View style={styles.summaryRow}>
                    <Text style={styles.summaryLabel}>Date:</Text>
                    <Text style={styles.summaryValue}>
                        {new Date(prescription.created_at?.toDate()).toLocaleDateString()}
                    </Text>
                </View>
                <View style={styles.amountContainer}>
                    <Text style={styles.amountLabel}>Amount Due:</Text>
                    <Text style={styles.amountValue}>₹{prescription.amount}</Text>
                </View>
            </View>

            {/* Payment Methods */}
            <View style={styles.paymentMethodsContainer}>
                <Text style={styles.sectionTitle}>Select Payment Method</Text>
                {paymentMethods.map((method) => (
                    <PaymentMethodCard
                        key={method.id}
                        method={method}
                        isSelected={selectedPaymentMethod?.id === method.id}
                        onSelect={handlePaymentMethodSelect}
                    />
                ))}
            </View>

            {/* Total Amount */}
            {selectedPaymentMethod && (
                <View style={styles.totalAmountContainer}>
                    <View style={styles.totalAmountRow}>
                        <Text style={styles.totalAmountLabel}>Prescription Amount:</Text>
                        <Text style={styles.totalAmountValue}>₹{prescription.amount}</Text>
                    </View>
                    {selectedPaymentMethod.processing_fee > 0 && (
                        <View style={styles.totalAmountRow}>
                            <Text style={styles.totalAmountLabel}>Processing Fee:</Text>
                            <Text style={styles.totalAmountValue}>₹{selectedPaymentMethod.processing_fee}</Text>
                        </View>
                    )}
                    <View style={[styles.totalAmountRow, styles.finalTotal]}>
                        <Text style={styles.finalTotalLabel}>Total Amount:</Text>
                        <Text style={styles.finalTotalValue}>₹{calculateTotalAmount()}</Text>
                    </View>
                </View>
            )}

            {/* Pay Button */}
            <TouchableOpacity
                style={[styles.payButton, !selectedPaymentMethod && styles.payButtonDisabled]}
                onPress={handlePaymentInitiation}
                disabled={!selectedPaymentMethod || loading}
            >
                <Text style={styles.payButtonText}>
                    {selectedPaymentMethod?.id === 'cash' ? 'Select Cash Payment' : 'Pay Now'}
                </Text>
            </TouchableOpacity>

            {/* Payment Security Info */}
            <View style={styles.securityInfo}>
                <Icon name="security" size={16} color="#666" />
                <Text style={styles.securityText}>
                    Your payment is secure and encrypted. All transactions are processed through Razorpay.
                </Text>
            </View>
        </ScrollView>
    );
};

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: '#f5f5f5',
        padding: 16,
    },
    loadingContainer: {
        flex: 1,
        justifyContent: 'center',
        alignItems: 'center',
        backgroundColor: '#f5f5f5',
    },
    loadingText: {
        marginTop: 16,
        fontSize: 16,
        color: '#666',
    },
    prescriptionSummary: {
        backgroundColor: 'white',
        borderRadius: 8,
        padding: 16,
        marginBottom: 16,
        elevation: 2,
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.1,
        shadowRadius: 4,
    },
    sectionTitle: {
        fontSize: 18,
        fontWeight: 'bold',
        color: '#333',
        marginBottom: 12,
    },
    summaryRow: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        marginBottom: 8,
    },
    summaryLabel: {
        fontSize: 14,
        color: '#666',
    },
    summaryValue: {
        fontSize: 14,
        color: '#333',
        fontWeight: '500',
    },
    amountContainer: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        marginTop: 12,
        paddingTop: 12,
        borderTopWidth: 1,
        borderTopColor: '#eee',
    },
    amountLabel: {
        fontSize: 16,
        fontWeight: 'bold',
        color: '#333',
    },
    amountValue: {
        fontSize: 18,
        fontWeight: 'bold',
        color: '#3399cc',
    },
    paymentMethodsContainer: {
        marginBottom: 16,
    },
    paymentMethodCard: {
        backgroundColor: 'white',
        borderRadius: 8,
        padding: 16,
        marginBottom: 12,
        borderWidth: 2,
        borderColor: 'transparent',
        elevation: 2,
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.1,
        shadowRadius: 4,
    },
    selectedPaymentMethod: {
        borderColor: '#3399cc',
        backgroundColor: '#f0f8ff',
    },
    paymentMethodHeader: {
        flexDirection: 'row',
        alignItems: 'flex-start',
    },
    paymentMethodInfo: {
        marginLeft: 12,
        flex: 1,
    },
    paymentMethodName: {
        fontSize: 16,
        fontWeight: 'bold',
        color: '#333',
        marginBottom: 4,
    },
    paymentMethodDescription: {
        fontSize: 14,
        color: '#666',
        lineHeight: 20,
    },
    paymentMethodNote: {
        fontSize: 12,
        color: '#3399cc',
        marginTop: 4,
        fontStyle: 'italic',
    },
    processingFeeContainer: {
        marginTop: 8,
        paddingTop: 8,
        borderTopWidth: 1,
        borderTopColor: '#eee',
    },
    processingFeeText: {
        fontSize: 12,
        color: '#ff6600',
        textAlign: 'right',
    },
    selectedIndicator: {
        position: 'absolute',
        top: 12,
        right: 12,
    },
    totalAmountContainer: {
        backgroundColor: 'white',
        borderRadius: 8,
        padding: 16,
        marginBottom: 16,
        elevation: 2,
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.1,
        shadowRadius: 4,
    },
    totalAmountRow: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        marginBottom: 8,
    },
    totalAmountLabel: {
        fontSize: 14,
        color: '#666',
    },
    totalAmountValue: {
        fontSize: 14,
        color: '#333',
        fontWeight: '500',
    },
    finalTotal: {
        marginTop: 8,
        paddingTop: 8,
        borderTopWidth: 1,
        borderTopColor: '#eee',
    },
    finalTotalLabel: {
        fontSize: 16,
        fontWeight: 'bold',
        color: '#333',
    },
    finalTotalValue: {
        fontSize: 18,
        fontWeight: 'bold',
        color: '#3399cc',
    },
    payButton: {
        backgroundColor: '#3399cc',
        borderRadius: 8,
        padding: 16,
        alignItems: 'center',
        marginBottom: 16,
    },
    payButtonDisabled: {
        backgroundColor: '#ccc',
    },
    payButtonText: {
        color: 'white',
        fontSize: 16,
        fontWeight: 'bold',
    },
    securityInfo: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'center',
        padding: 12,
    },
    securityText: {
        fontSize: 12,
        color: '#666',
        marginLeft: 8,
        textAlign: 'center',
        flex: 1,
    },
});

export default PatientPaymentScreen;
