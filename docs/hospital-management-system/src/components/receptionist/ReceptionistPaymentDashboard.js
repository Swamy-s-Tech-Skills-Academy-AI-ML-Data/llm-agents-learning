/**
 * Receptionist Payment Dashboard Component
 * Phase 2: Cash payment processing and pending payment management
 */

import React, { useState, useEffect } from 'react';
import {
    View,
    Text,
    StyleSheet,
    FlatList,
    TouchableOpacity,
    Alert,
    ActivityIndicator,
    TextInput,
    Modal,
    RefreshControl
} from 'react-native';
import Icon from 'react-native-vector-icons/MaterialIcons';
import { collection, query, where, onSnapshot, orderBy } from 'firebase/firestore';
import { db } from '../../config/firebase';
import { httpsCallable } from 'firebase/functions';
import { functions } from '../../config/firebase';

const ReceptionistPaymentDashboard = ({ navigation }) => {
    const [pendingPayments, setPendingPayments] = useState([]);
    const [loading, setLoading] = useState(true);
    const [refreshing, setRefreshing] = useState(false);
    const [showPaymentModal, setShowPaymentModal] = useState(false);
    const [selectedPrescription, setSelectedPrescription] = useState(null);
    const [processingPayment, setProcessingPayment] = useState(false);
    const [searchQuery, setSearchQuery] = useState('');
    const [filteredPayments, setFilteredPayments] = useState([]);

    useEffect(() => {
        const unsubscribe = subscribeToPayments();
        return () => unsubscribe && unsubscribe();
    }, []);

    useEffect(() => {
        filterPayments();
    }, [pendingPayments, searchQuery]);

    const subscribeToPayments = () => {
        try {
            // Query for prescriptions with pending cash payments or unpaid status
            const paymentsQuery = query(
                collection(db, 'prescriptions'),
                where('payment_status', 'in', ['pending', 'pending_cash', 'failed']),
                orderBy('created_at', 'desc')
            );

            const unsubscribe = onSnapshot(paymentsQuery, (snapshot) => {
                const payments = snapshot.docs.map(doc => ({
                    id: doc.id,
                    ...doc.data()
                }));
                setPendingPayments(payments);
                setLoading(false);
            });

            return unsubscribe;
        } catch (error) {
            console.error('Error subscribing to payments:', error);
            setLoading(false);
            return null;
        }
    };

    const filterPayments = () => {
        if (!searchQuery.trim()) {
            setFilteredPayments(pendingPayments);
            return;
        }

        const filtered = pendingPayments.filter(payment =>
            payment.patient_name?.toLowerCase().includes(searchQuery.toLowerCase()) ||
            payment.patient_phone?.includes(searchQuery) ||
            payment.id.toLowerCase().includes(searchQuery.toLowerCase()) ||
            payment.doctor_name?.toLowerCase().includes(searchQuery.toLowerCase())
        );

        setFilteredPayments(filtered);
    };

    const handleRefresh = async () => {
        setRefreshing(true);
        // The real-time listener will automatically update the data
        setTimeout(() => setRefreshing(false), 1000);
    };

    const handleCashPayment = (prescription) => {
        setSelectedPrescription(prescription);
        setShowPaymentModal(true);
    };

    const confirmCashPayment = async () => {
        if (!selectedPrescription) return;

        try {
            setProcessingPayment(true);

            const updateCashPaymentStatus = httpsCallable(functions, 'updateCashPaymentStatus');
            const result = await updateCashPaymentStatus({
                prescriptionId: selectedPrescription.id,
                amount: selectedPrescription.amount,
                patient_id: selectedPrescription.patient_id
            });

            if (result.data.success) {
                Alert.alert('Success', 'Cash payment has been recorded successfully');
                setShowPaymentModal(false);
                setSelectedPrescription(null);
            } else {
                throw new Error(result.data.message || 'Failed to update payment status');
            }
        } catch (error) {
            console.error('Error processing cash payment:', error);
            Alert.alert('Error', 'Failed to process cash payment. Please try again.');
        } finally {
            setProcessingPayment(false);
        }
    };

    const getPaymentStatusColor = (status) => {
        const colorMap = {
            'pending': '#ff9800',
            'pending_cash': '#2196f3',
            'failed': '#f44336',
            'processing': '#9c27b0'
        };
        return colorMap[status] || '#666';
    };

    const getPaymentStatusText = (status) => {
        const textMap = {
            'pending': 'Payment Due',
            'pending_cash': 'Cash Payment Selected',
            'failed': 'Payment Failed',
            'processing': 'Processing'
        };
        return textMap[status] || status;
    };

    const PaymentItem = ({ item }) => (
        <View style={styles.paymentItem}>
            {/* Patient Info */}
            <View style={styles.paymentHeader}>
                <View style={styles.patientInfo}>
                    <Text style={styles.patientName}>{item.patient_name || 'Unknown Patient'}</Text>
                    <Text style={styles.patientPhone}>{item.patient_phone || 'No phone'}</Text>
                </View>
                <View style={styles.statusContainer}>
                    <View style={[styles.statusBadge, { backgroundColor: getPaymentStatusColor(item.payment_status) }]}>
                        <Text style={styles.statusText}>{getPaymentStatusText(item.payment_status)}</Text>
                    </View>
                </View>
            </View>

            {/* Prescription Details */}
            <View style={styles.prescriptionDetails}>
                <View style={styles.detailRow}>
                    <Icon name="assignment" size={16} color="#666" />
                    <Text style={styles.detailText}>ID: {item.id}</Text>
                </View>
                <View style={styles.detailRow}>
                    <Icon name="person" size={16} color="#666" />
                    <Text style={styles.detailText}>Dr. {item.doctor_name || 'Unknown'}</Text>
                </View>
                <View style={styles.detailRow}>
                    <Icon name="schedule" size={16} color="#666" />
                    <Text style={styles.detailText}>
                        {item.created_at ? new Date(item.created_at.toDate()).toLocaleDateString() : 'No date'}
                    </Text>
                </View>
            </View>

            {/* Amount and Actions */}
            <View style={styles.paymentFooter}>
                <View style={styles.amountContainer}>
                    <Text style={styles.amountLabel}>Amount Due:</Text>
                    <Text style={styles.amountValue}>₹{item.amount || 0}</Text>
                </View>

                {item.payment_status === 'pending_cash' && (
                    <TouchableOpacity
                        style={styles.cashPaymentButton}
                        onPress={() => handleCashPayment(item)}
                    >
                        <Icon name="money" size={20} color="white" />
                        <Text style={styles.cashPaymentButtonText}>Confirm Cash Payment</Text>
                    </TouchableOpacity>
                )}

                {item.payment_status === 'failed' && (
                    <TouchableOpacity
                        style={styles.retryButton}
                        onPress={() => {/* Navigate to retry payment */ }}
                    >
                        <Icon name="refresh" size={20} color="white" />
                        <Text style={styles.retryButtonText}>Request Retry</Text>
                    </TouchableOpacity>
                )}
            </View>
        </View>
    );

    const PaymentConfirmationModal = () => (
        <Modal
            visible={showPaymentModal}
            transparent={true}
            animationType="slide"
            onRequestClose={() => setShowPaymentModal(false)}
        >
            <View style={styles.modalOverlay}>
                <View style={styles.modalContainer}>
                    <Text style={styles.modalTitle}>Confirm Cash Payment</Text>

                    {selectedPrescription && (
                        <View style={styles.modalContent}>
                            <Text style={styles.modalText}>Patient: {selectedPrescription.patient_name}</Text>
                            <Text style={styles.modalText}>Prescription ID: {selectedPrescription.id}</Text>
                            <Text style={styles.modalText}>Amount: ₹{selectedPrescription.amount}</Text>

                            <Text style={styles.confirmationText}>
                                Confirm that you have received ₹{selectedPrescription.amount} in cash from the patient?
                            </Text>
                        </View>
                    )}

                    <View style={styles.modalButtons}>
                        <TouchableOpacity
                            style={styles.cancelButton}
                            onPress={() => setShowPaymentModal(false)}
                            disabled={processingPayment}
                        >
                            <Text style={styles.cancelButtonText}>Cancel</Text>
                        </TouchableOpacity>

                        <TouchableOpacity
                            style={styles.confirmButton}
                            onPress={confirmCashPayment}
                            disabled={processingPayment}
                        >
                            {processingPayment ? (
                                <ActivityIndicator size="small" color="white" />
                            ) : (
                                <Text style={styles.confirmButtonText}>Confirm Payment</Text>
                            )}
                        </TouchableOpacity>
                    </View>
                </View>
            </View>
        </Modal>
    );

    if (loading) {
        return (
            <View style={styles.loadingContainer}>
                <ActivityIndicator size="large" color="#3399cc" />
                <Text style={styles.loadingText}>Loading payments...</Text>
            </View>
        );
    }

    return (
        <View style={styles.container}>
            {/* Header */}
            <View style={styles.header}>
                <Text style={styles.headerTitle}>Payment Management</Text>
                <View style={styles.headerStats}>
                    <Text style={styles.statsText}>
                        {filteredPayments.length} pending payment{filteredPayments.length !== 1 ? 's' : ''}
                    </Text>
                </View>
            </View>

            {/* Search Bar */}
            <View style={styles.searchContainer}>
                <Icon name="search" size={20} color="#666" style={styles.searchIcon} />
                <TextInput
                    style={styles.searchInput}
                    placeholder="Search by patient name, phone, or prescription ID..."
                    value={searchQuery}
                    onChangeText={setSearchQuery}
                />
                {searchQuery.length > 0 && (
                    <TouchableOpacity
                        onPress={() => setSearchQuery('')}
                        style={styles.clearSearchButton}
                    >
                        <Icon name="clear" size={20} color="#666" />
                    </TouchableOpacity>
                )}
            </View>

            {/* Payment List */}
            {filteredPayments.length === 0 ? (
                <View style={styles.emptyContainer}>
                    <Icon name="payment" size={64} color="#ccc" />
                    <Text style={styles.emptyText}>
                        {searchQuery ? 'No payments found matching your search' : 'No pending payments'}
                    </Text>
                </View>
            ) : (
                <FlatList
                    data={filteredPayments}
                    keyExtractor={(item) => item.id}
                    renderItem={({ item }) => <PaymentItem item={item} />}
                    contentContainerStyle={styles.listContainer}
                    refreshControl={
                        <RefreshControl
                            refreshing={refreshing}
                            onRefresh={handleRefresh}
                            colors={['#3399cc']}
                        />
                    }
                />
            )}

            <PaymentConfirmationModal />
        </View>
    );
};

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: '#f5f5f5',
    },
    loadingContainer: {
        flex: 1,
        justifyContent: 'center',
        alignItems: 'center',
    },
    loadingText: {
        marginTop: 16,
        fontSize: 16,
        color: '#666',
    },
    header: {
        backgroundColor: 'white',
        padding: 16,
        elevation: 2,
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.1,
        shadowRadius: 4,
    },
    headerTitle: {
        fontSize: 20,
        fontWeight: 'bold',
        color: '#333',
    },
    headerStats: {
        marginTop: 4,
    },
    statsText: {
        fontSize: 14,
        color: '#666',
    },
    searchContainer: {
        flexDirection: 'row',
        alignItems: 'center',
        backgroundColor: 'white',
        margin: 16,
        borderRadius: 8,
        paddingHorizontal: 12,
        elevation: 1,
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 1 },
        shadowOpacity: 0.1,
        shadowRadius: 2,
    },
    searchIcon: {
        marginRight: 8,
    },
    searchInput: {
        flex: 1,
        paddingVertical: 12,
        fontSize: 16,
        color: '#333',
    },
    clearSearchButton: {
        padding: 4,
    },
    listContainer: {
        padding: 16,
    },
    paymentItem: {
        backgroundColor: 'white',
        borderRadius: 8,
        padding: 16,
        marginBottom: 12,
        elevation: 2,
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.1,
        shadowRadius: 4,
    },
    paymentHeader: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'flex-start',
        marginBottom: 12,
    },
    patientInfo: {
        flex: 1,
    },
    patientName: {
        fontSize: 16,
        fontWeight: 'bold',
        color: '#333',
    },
    patientPhone: {
        fontSize: 14,
        color: '#666',
        marginTop: 2,
    },
    statusContainer: {
        alignItems: 'flex-end',
    },
    statusBadge: {
        paddingHorizontal: 8,
        paddingVertical: 4,
        borderRadius: 4,
    },
    statusText: {
        color: 'white',
        fontSize: 12,
        fontWeight: 'bold',
    },
    prescriptionDetails: {
        marginBottom: 12,
    },
    detailRow: {
        flexDirection: 'row',
        alignItems: 'center',
        marginBottom: 6,
    },
    detailText: {
        fontSize: 14,
        color: '#666',
        marginLeft: 8,
    },
    paymentFooter: {
        borderTopWidth: 1,
        borderTopColor: '#eee',
        paddingTop: 12,
    },
    amountContainer: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        marginBottom: 12,
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
    cashPaymentButton: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'center',
        backgroundColor: '#4CAF50',
        borderRadius: 6,
        padding: 12,
    },
    cashPaymentButtonText: {
        color: 'white',
        fontSize: 14,
        fontWeight: 'bold',
        marginLeft: 8,
    },
    retryButton: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'center',
        backgroundColor: '#ff9800',
        borderRadius: 6,
        padding: 12,
    },
    retryButtonText: {
        color: 'white',
        fontSize: 14,
        fontWeight: 'bold',
        marginLeft: 8,
    },
    emptyContainer: {
        flex: 1,
        justifyContent: 'center',
        alignItems: 'center',
        padding: 32,
    },
    emptyText: {
        fontSize: 16,
        color: '#666',
        textAlign: 'center',
        marginTop: 16,
    },
    modalOverlay: {
        flex: 1,
        backgroundColor: 'rgba(0, 0, 0, 0.5)',
        justifyContent: 'center',
        alignItems: 'center',
    },
    modalContainer: {
        backgroundColor: 'white',
        borderRadius: 8,
        padding: 24,
        margin: 32,
        minWidth: 300,
    },
    modalTitle: {
        fontSize: 18,
        fontWeight: 'bold',
        color: '#333',
        marginBottom: 16,
        textAlign: 'center',
    },
    modalContent: {
        marginBottom: 24,
    },
    modalText: {
        fontSize: 14,
        color: '#666',
        marginBottom: 8,
    },
    confirmationText: {
        fontSize: 16,
        color: '#333',
        marginTop: 12,
        lineHeight: 24,
    },
    modalButtons: {
        flexDirection: 'row',
        justifyContent: 'space-between',
    },
    cancelButton: {
        flex: 1,
        backgroundColor: '#ccc',
        borderRadius: 6,
        padding: 12,
        marginRight: 8,
        alignItems: 'center',
    },
    cancelButtonText: {
        color: '#333',
        fontSize: 14,
        fontWeight: 'bold',
    },
    confirmButton: {
        flex: 1,
        backgroundColor: '#4CAF50',
        borderRadius: 6,
        padding: 12,
        marginLeft: 8,
        alignItems: 'center',
    },
    confirmButtonText: {
        color: 'white',
        fontSize: 14,
        fontWeight: 'bold',
    },
});

export default ReceptionistPaymentDashboard;
